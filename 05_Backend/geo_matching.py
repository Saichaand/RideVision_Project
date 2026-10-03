"""
Backend geo-matching and complaint-routing logic for RideVision.

Features:
1. Warning check — detects if an active, confirmed pothole is within alert distance (default 250m)
   and aligned with vehicle travel direction (+/- 45 degrees).
2. Passed-by matching — after a trip ends, identifies potholes the route passed within 25m of,
   prompting the commuter for "still there" or "fixed" confirmation.
3. Confirmation cooldown — 7-day cooldown per user per pothole to eliminate duplicate/spam confirmations.
4. City-based complaint routing — auto-identifies city jurisdiction (with offline geo-lookup for
   Mangaluru, Bengaluru, Udupi, and Mysuru) and routes complaints to appropriate authorities
   (MCC WhatsApp, BBMP Helpline, Email).
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from math import radians, sin, cos, sqrt, atan2, degrees
from typing import Optional, List, Dict, Any
from urllib.parse import quote
import os
import requests
import xml.etree.ElementTree as ET

EARTH_RADIUS_M = 6371000
GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY", "")


@dataclass
class Pothole:
    id: str
    lat: float
    lon: float
    city: str
    severity: str  # minor | moderate | severe
    status: str  # active | reported_fixed | verified_fixed
    confirmation_count: int
    address: Optional[str] = None


@dataclass
class TripPoint:
    lat: float
    lon: float
    heading: Optional[float] = None  # degrees, 0 = north, clockwise
    speed_kmh: Optional[float] = None
    timestamp: Optional[str] = None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the great-circle distance between two GPS points in meters."""
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates compass bearing from point 1 to point 2, in degrees (0-360, 0 = North)."""
    phi1, phi2 = radians(lat1), radians(lat2)
    d_lambda = radians(lon2 - lon1)

    x = sin(d_lambda) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(d_lambda)
    return (degrees(atan2(x, y)) + 360) % 360


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two compass bearings in degrees (0-180)."""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


def bounding_box(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Generates a rough lat/lon bounding box around a coordinate for O(1) database pre-filtering."""
    lat_delta = radius_m / 111_000.0
    lon_delta = radius_m / (111_000.0 * max(cos(radians(lat)), 0.1))
    return (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta)


def get_nearby_potholes(
    current_lat: float,
    current_lon: float,
    radius_m: float,
    all_potholes: List[Pothole]
) -> List[Pothole]:
    """Pre-filters candidate potholes by bounding box, then applies exact Haversine distance."""
    min_lat, max_lat, min_lon, max_lon = bounding_box(current_lat, current_lon, radius_m)

    candidates = [
        p for p in all_potholes
        if min_lat <= p.lat <= max_lat and min_lon <= p.lon <= max_lon
    ]

    return [
        p for p in candidates
        if haversine_distance_m(current_lat, current_lon, p.lat, p.lon) <= radius_m
    ]


def check_warning_ahead(
    current_lat: float,
    current_lon: float,
    heading_deg: float,
    all_potholes: List[Pothole],
    alert_radius_m: float = 250.0,
    heading_tolerance_deg: float = 45.0,
    min_confirmation_count: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Checks if there is an active, confirmed pothole ahead of the driver within
    the alert radius and inside the directional field-of-view cone (+/- heading_tolerance_deg).
    Returns the nearest hazard warning or None.
    """
    nearby = get_nearby_potholes(current_lat, current_lon, alert_radius_m, all_potholes)

    qualifying = []
    for p in nearby:
        if p.status != "active":
            continue
        if p.confirmation_count < min_confirmation_count:
            continue

        pothole_bearing = bearing_deg(current_lat, current_lon, p.lat, p.lon)
        deviation = angle_diff_deg(heading_deg, pothole_bearing)
        if deviation > heading_tolerance_deg:
            continue

        distance = haversine_distance_m(current_lat, current_lon, p.lat, p.lon)
        qualifying.append((distance, deviation, p))

    if not qualifying:
        return None

    distance, dev, nearest = min(qualifying, key=lambda x: x[0])

    # Severity urgency level
    urgency = "high" if nearest.severity == "severe" or distance < 80 else "moderate"

    return {
        "pothole_id": nearest.id,
        "distance_m": round(distance),
        "angular_deviation_deg": round(dev, 1),
        "severity": nearest.severity,
        "urgency": urgency,
        "address": nearest.address or f"Coordinates: {nearest.lat:.4f}, {nearest.lon:.4f}",
        "confirmation_count": nearest.confirmation_count,
        "message": f"Caution: {nearest.severity.capitalize()} pothole {round(distance)}m ahead!"
    }


def match_trip_passed_by(
    trip_points: List[TripPoint],
    all_potholes: List[Pothole],
    proximity_radius_m: float = 25.0
) -> List[Dict[str, Any]]:
    """
    After a commute/trip concludes, compares the vehicle's breadcrumb path against
    all registered potholes. Returns potholes the vehicle passed within proximity_radius_m.
    """
    passed_map: Dict[str, Dict[str, Any]] = {}

    for point in trip_points:
        nearby = get_nearby_potholes(point.lat, point.lon, proximity_radius_m, all_potholes)
        for p in nearby:
            dist = haversine_distance_m(point.lat, point.lon, p.lat, p.lon)
            if p.id not in passed_map or dist < passed_map[p.id]["closest_distance_m"]:
                passed_map[p.id] = {
                    "pothole_id": p.id,
                    "address": p.address,
                    "severity": p.severity,
                    "status": p.status,
                    "closest_distance_m": round(dist, 1)
                }

    return list(passed_map.values())


def is_confirmation_eligible(
    user_id: str,
    pothole_id: str,
    last_confirmation_at: Optional[datetime],
    cooldown_days: int = 7
) -> bool:
    """
    Enforces anti-gaming cooldown: a user may only confirm a pothole once every 7 days.
    """
    if last_confirmation_at is None:
        return True
    
    now = datetime.now(timezone.utc)
    if last_confirmation_at.tzinfo is None:
        last_confirmation_at = last_confirmation_at.replace(tzinfo=timezone.utc)
        
    return (now - last_confirmation_at) > timedelta(days=cooldown_days)


# --- City Identification & Municipal Dispatch ---

# City center bounding coordinates for robust offline geocoding
CITY_BOUNDS = [
    {
        "name": "Mangaluru",
        "lat_min": 12.8000, "lat_max": 13.0200,
        "lon_min": 74.8000, "lon_max": 74.9600
    },
    {
        "name": "Bengaluru",
        "lat_min": 12.8000, "lat_max": 13.1500,
        "lon_min": 77.4500, "lon_max": 77.7800
    },
    {
        "name": "Udupi",
        "lat_min": 13.2500, "lat_max": 13.4500,
        "lon_min": 74.7000, "lon_max": 74.8500
    },
    {
        "name": "Mysuru",
        "lat_min": 12.2500, "lat_max": 12.4000,
        "lon_min": 76.5500, "lon_max": 76.7500
    }
]


def get_city_from_coords(lat: float, lon: float) -> str:
    """
    Determines city jurisdiction from coordinates.
    1. First checks offline polygon boundaries (instant, zero cost, reliable).
    2. Falls back to reverse geocoding via OpenStreetMap Nominatim.
    3. Falls back to Google Geocoding API if key is available.
    """
    for city in CITY_BOUNDS:
        if city["lat_min"] <= lat <= city["lat_max"] and city["lon_min"] <= lon <= city["lon_max"]:
            return city["name"]

    # Try reverse geocode to detect city/district
    try:
        geo = reverse_geocode(lat, lon)
        city_cand = geo.get("city")
        if city_cand and city_cand != "Unknown":
            return city_cand
    except Exception:
        pass

    if GEOCODING_API_KEY:
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"latlng": f"{lat},{lon}", "key": GEOCODING_API_KEY},
                timeout=3
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    for comp in results[0].get("address_components", []):
                        if "locality" in comp.get("types", []):
                            return comp["long_name"]
        except Exception:
            pass

    return "Mangaluru"


def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """
    Reverse geocodes (lat, lon) to human-readable road name, suburb, city, and state
    using OpenStreetMap Nominatim with offline fallback.
    """
    # 1. Quick city detection via bounds
    offline_city = "Mangaluru"
    for c in CITY_BOUNDS:
        if c["lat_min"] <= lat <= c["lat_max"] and c["lon_min"] <= lon <= c["lon_max"]:
            offline_city = c["name"]
            break

    result = {
        "road": f"Road at {lat:.5f}, {lon:.5f}",
        "city": offline_city,
        "state": "Karnataka",
        "display_name": f"{lat:.5f}, {lon:.5f} ({offline_city})"
    }

    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=18&addressdetails=1"
        headers = {"User-Agent": "RideVision-PBL-RoadSafetyApp/1.0 (contact: student@sjec.ac.in)"}
        resp = requests.get(url, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            addr = data.get("address", {})

            # Standardize city name
            raw_city = (
                addr.get("city") or
                addr.get("town") or
                addr.get("municipality") or
                addr.get("district") or
                addr.get("county") or
                offline_city
            )

            # Clean and normalize common Karnataka city names
            cleaned_city = raw_city
            lower_city = raw_city.lower()
            if "mangal" in lower_city:
                cleaned_city = "Mangaluru"
            elif "bengal" in lower_city or "bangal" in lower_city:
                cleaned_city = "Bengaluru"
            elif "udupi" in lower_city or "manipal" in lower_city:
                cleaned_city = "Udupi"
            elif "mys" in lower_city:
                cleaned_city = "Mysuru"

            road_name = (
                addr.get("road") or
                addr.get("highway") or
                addr.get("suburb") or
                addr.get("neighbourhood") or
                data.get("name") or
                f"Road near {cleaned_city}"
            )

            result["road"] = road_name
            result["city"] = cleaned_city
            result["state"] = addr.get("state", "Karnataka")
            result["display_name"] = data.get("display_name", f"{road_name}, {cleaned_city}")
    except Exception:
        pass

    return result


def geocode_address(query: str) -> Optional[Dict[str, Any]]:
    """
    Forward geocodes an address or landmark query (e.g. 'SJEC Vamanjoor', 'Hampankatta Mangaluru')
    to (lat, lon, display_name, city).
    """
    if not query or len(query.strip()) < 2:
        return None

    try:
        url = f"https://nominatim.openstreetmap.org/search?q={quote(query.strip())}&format=json&limit=1&addressdetails=1"
        headers = {"User-Agent": "RideVision-PBL-RoadSafetyApp/1.0 (contact: student@sjec.ac.in)"}
        resp = requests.get(url, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            results = resp.json()
            if results and len(results) > 0:
                top = results[0]
                lat = float(top["lat"])
                lon = float(top["lon"])
                addr = top.get("address", {})
                city = get_city_from_coords(lat, lon)
                road = addr.get("road") or addr.get("suburb") or top.get("display_name", "").split(",")[0]
                return {
                    "lat": lat,
                    "lon": lon,
                    "city": city,
                    "road": road,
                    "display_name": top.get("display_name")
                }
    except Exception:
        pass

    return None


def detect_ip_location() -> Dict[str, Any]:
    """
    Detects user's real-time geographical coordinates based on public IP address.
    Zero-config, fast, no browser permission prompt required.
    """
    try:
        resp = requests.get("https://ipapi.co/json/", headers={"User-Agent": "RideVision/1.0"}, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            lat = float(data.get("latitude", 12.9152))
            lon = float(data.get("longitude", 74.8988))
            city = data.get("city", "Mangaluru")
            region = data.get("region", "Karnataka")
            return {
                "lat": lat,
                "lon": lon,
                "city": city,
                "region": region,
                "isp": data.get("org", "Local ISP"),
                "source": "ip_geolocation"
            }
    except Exception:
        pass

    # Default to Mangaluru (SJEC Vamanjoor)
    return {
        "lat": 12.9152,
        "lon": 74.8988,
        "city": "Mangaluru",
        "region": "Karnataka",
        "isp": "Default Network",
        "source": "default"
    }


def get_route_osrm(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float
) -> Dict[str, Any]:
    """
    Computes real-world driving route between origin and destination using the OSRM Routing Engine.
    Returns:
    - coordinates: List of [lat, lon] waypoints along the actual road network
    - distance_km: Real road distance in kilometers
    - duration_min: Estimated drive duration in minutes
    - steps: Major turn-by-turn road steps
    """
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{start_lon},{start_lat};{end_lon},{end_lat}"
            f"?overview=full&geometries=geojson&steps=true"
        )
        resp = requests.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                # GeoJSON coordinates are [lon, lat] -> convert to [lat, lon]
                coords = [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]]
                dist_km = round(route["distance"] / 1000.0, 2)
                dur_min = round(route["duration"] / 60.0, 1)

                steps = []
                for leg in route.get("legs", []):
                    for step in leg.get("steps", []):
                        name = step.get("name") or "Connecting Road"
                        dist = round(step.get("distance", 0))
                        steps.append({"name": name, "distance_m": dist})

                return {
                    "status": "success",
                    "coordinates": coords,
                    "distance_km": dist_km,
                    "duration_min": dur_min,
                    "steps": steps,
                    "source": "osrm_live"
                }
    except Exception:
        pass

    # Fallback: compute 25 linearly interpolated points if OSRM is unreachable
    coords = []
    num_pts = 25
    for i in range(num_pts):
        f = i / (num_pts - 1)
        coords.append([
            start_lat + f * (end_lat - start_lat),
            start_lon + f * (end_lon - start_lon)
        ])
    straight_dist_km = round(haversine_distance_m(start_lat, start_lon, end_lat, end_lon) / 1000.0, 2)

    return {
        "status": "fallback",
        "coordinates": coords,
        "distance_km": straight_dist_km,
        "duration_min": round(straight_dist_km / 35.0 * 60, 1),
        "steps": [{"name": "Direct Commute Corridor", "distance_m": int(straight_dist_km * 1000)}],
        "source": "interpolated_corridor"
    }


def parse_gpx_content(gpx_text_or_bytes: Any) -> Dict[str, Any]:
    """
    Parses a GPX (GPS Exchange Format) XML string or byte stream.
    Extracts ordered track coordinates (lat, lon), elevation, timestamps, and calculates distance.
    Supports <trkpt>, <rtept>, and <wpt> tags with or without XML namespaces.
    """
    if isinstance(gpx_text_or_bytes, bytes):
        gpx_text = gpx_text_or_bytes.decode("utf-8", errors="ignore")
    else:
        gpx_text = str(gpx_text_or_bytes)

    root = ET.fromstring(gpx_text)

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    track_name = "GPX Commute Route"
    for elem in root.iter():
        if strip_ns(elem.tag) == "name" and elem.text:
            track_name = elem.text.strip()
            break

    coords: List[List[float]] = []
    elevations: List[float] = []
    timestamps: List[str] = []

    for elem in root.iter():
        tag = strip_ns(elem.tag)
        if tag in ("trkpt", "rtept", "wpt"):
            lat = elem.attrib.get("lat")
            lon = elem.attrib.get("lon")
            if lat is not None and lon is not None:
                coords.append([float(lat), float(lon)])
                for child in elem:
                    c_tag = strip_ns(child.tag)
                    if c_tag == "ele" and child.text:
                        try:
                            elevations.append(float(child.text))
                        except ValueError:
                            pass
                    elif c_tag == "time" and child.text:
                        timestamps.append(child.text.strip())

    if not coords:
        raise ValueError("No valid GPS trackpoints (<trkpt>, <rtept>, <wpt>) found in GPX file.")

    # Calculate cumulative distance along path using Haversine formulation
    total_dist_m = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        total_dist_m += haversine_distance_m(p1[0], p1[1], p2[0], p2[1])

    dist_km = round(total_dist_m / 1000.0, 2)
    est_dur_min = round(max(1.0, (dist_km / 35.0) * 60.0), 1)

    return {
        "status": "success",
        "name": track_name,
        "coordinates": coords,
        "distance_km": dist_km,
        "duration_min": est_dur_min,
        "total_points": len(coords),
        "elevations": elevations,
        "timestamps": timestamps,
        "source": "gpx_track"
    }


def load_gpx_file(file_path: str) -> Dict[str, Any]:
    """Reads a .gpx or .csv GPS log file from local disk and parses its coordinates."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if file_path.lower().endswith(".csv") or not ("<gpx" in content or "<?xml" in content):
        return parse_csv_gps_log(content, track_name=os.path.basename(file_path))
    return parse_gpx_content(content)


def parse_csv_gps_log(csv_text_or_bytes: Any, track_name: str = "Recorded GPS Commute Log") -> Dict[str, Any]:
    """
    Parses CSV-formatted GPS logs (e.g. from mobile logger, GPS logger, OsmAnd, or custom tracker).
    Extracts ordered coordinates (lat, lon), elevation, speed, bearing, and timestamps.
    """
    import csv
    import io

    if isinstance(csv_text_or_bytes, bytes):
        text = csv_text_or_bytes.decode("utf-8", errors="ignore")
    else:
        text = str(csv_text_or_bytes)

    reader = csv.DictReader(io.StringIO(text))
    coords: List[List[float]] = []
    elevations: List[float] = []
    timestamps: List[str] = []
    speeds: List[float] = []
    bearings: List[float] = []

    for row in reader:
        lat_val = row.get("lat") or row.get("latitude") or row.get("Latitude")
        lon_val = row.get("lon") or row.get("longitude") or row.get("Longitude")
        if lat_val and lon_val:
            try:
                coords.append([float(lat_val), float(lon_val)])
                if "elevation" in row and row["elevation"]:
                    elevations.append(float(row["elevation"]))
                if "time" in row and row["time"]:
                    timestamps.append(row["time"].strip())
                if "speed" in row and row["speed"]:
                    speeds.append(float(row["speed"]))
                if "bearing" in row and row["bearing"]:
                    bearings.append(float(row["bearing"]))
            except ValueError:
                continue

    if not coords:
        raise ValueError("No valid GPS latitude/longitude rows found in CSV data.")

    total_dist_m = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        total_dist_m += haversine_distance_m(p1[0], p1[1], p2[0], p2[1])

    dist_km = round(total_dist_m / 1000.0, 2)
    est_dur_min = round(max(1.0, (dist_km / 35.0) * 60.0), 1)

    return {
        "status": "success",
        "name": track_name,
        "coordinates": coords,
        "distance_km": dist_km,
        "duration_min": est_dur_min,
        "total_points": len(coords),
        "elevations": elevations,
        "timestamps": timestamps,
        "speeds": speeds,
        "bearings": bearings,
        "source": "gps_csv_log"
    }


def parse_gps_track(content_or_bytes: Any, name: Optional[str] = None) -> Dict[str, Any]:
    """Auto-detects track format (GPX XML vs CSV log) and parses into structured route data."""
    sample = content_or_bytes[:100] if isinstance(content_or_bytes, bytes) else str(content_or_bytes)[:100]
    if "<gpx" in sample or "<?xml" in sample:
        return parse_gpx_content(content_or_bytes)
    else:
        return parse_csv_gps_log(content_or_bytes, track_name=name or "Recorded GPS Commute Log")


def build_complaint_text(complaint: Dict[str, Any]) -> str:
    """Generates structured, professional complaint message text for WhatsApp or Email."""
    lat = complaint.get("lat") or complaint.get("latitude", 0.0)
    lon = complaint.get("lon") or complaint.get("longitude", 0.0)
    severity = str(complaint.get("severity", "Moderate")).upper()
    address = complaint.get("address", f"Lat: {lat:.5f}, Lon: {lon:.5f}")
    time_str = complaint.get("reported_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    note = complaint.get("note", "Reported by commuter via RideVision AI")

    return (
        f"🚨 *ROAD POTHOLE HAZARD REPORT — RideVision*\n\n"
        f"📍 *Location:* {address}\n"
        f"🌐 *Google Maps Link:* https://maps.google.com/?q={lat},{lon}\n"
        f"⚠️ *Severity Assessment:* {severity}\n"
        f"🕒 *Reported At:* {time_str}\n"
        f"📝 *Commuter Note:* {note}\n\n"
        f"_This report was verified and auto-routed via the RideVision Computer Vision Road Safety System._"
    )


def route_complaint(complaint: Dict[str, Any], city_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates actionable routing details based on city authority channel:
    - WhatsApp: direct clickable link (wa.me) with prefilled complaint text.
    - Helpline: phone number and instructions.
    - Email: mailto link with subject and body.
    Includes smart fallback to Municipal Corporation / PWD grievance desk.
    """
    city_name = city_config.get("city_name") or complaint.get("city") or "Mangaluru"
    channel = city_config.get("channel_type", "whatsapp")
    contact = city_config.get("contact_value", "")
    authority = city_config.get("authority_name") or f"{city_name} Municipal Corporation / PWD Cell"

    # Default contact fallback if empty
    if not contact:
        if "mangal" in city_name.lower():
            channel = "whatsapp"
            contact = "919449007722"
            authority = "Mangaluru City Corporation (MCC) Grievance Desk"
        elif "bengal" in city_name.lower() or "bangal" in city_name.lower():
            channel = "helpline"
            contact = "080-22660000"
            authority = "BBMP Pothole Control Room (24x7)"
        elif "udupi" in city_name.lower():
            channel = "helpline"
            contact = "0820-2520306"
            authority = "Udupi City Municipal Council (CMC)"
        elif "mys" in city_name.lower():
            channel = "helpline"
            contact = "0821-2440890"
            authority = "Mysuru City Corporation (MCC)"
        else:
            channel = "whatsapp"
            contact = "919449007722"
            authority = f"{city_name} Road Safety & PWD Grievance Desk"

    message = build_complaint_text(complaint)

    result = {
        "city": city_name,
        "authority": authority,
        "channel": channel,
        "instructions": city_config.get("instructions") or f"Direct complaint channel for {city_name}.",
        "action": "store_only",
        "link": None,
        "contact": contact,
        "prefilled_message": message
    }

    if channel == "whatsapp":
        result["action"] = "manual_forward"
        clean_number = contact.replace("+", "").replace("-", "").replace(" ", "")
        result["link"] = f"https://wa.me/{clean_number}?text={quote(message)}"

    elif channel == "email":
        result["action"] = "manual_forward"
        subject = quote(f"Road Pothole Hazard Report - {city_name}")
        body = quote(message)
        result["link"] = f"mailto:{contact}?subject={subject}&body={body}"

    elif channel == "helpline":
        result["action"] = "call_helpline"
        result["link"] = f"tel:{contact}"

    return result
