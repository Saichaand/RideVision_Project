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
    First checks offline polygon boundaries (instant, zero cost, reliable).
    Falls back to Google Geocoding API if key is available.
    """
    for city in CITY_BOUNDS:
        if city["lat_min"] <= lat <= city["lat_max"] and city["lon_min"] <= lon <= city["lon_max"]:
            return city["name"]

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

    return "Unknown"


def build_complaint_text(complaint: Dict[str, Any]) -> str:
    """Generates structured, professional complaint message text for WhatsApp or Email."""
    lat = complaint.get("lat") or complaint.get("latitude", 0.0)
    lon = complaint.get("lon") or complaint.get("longitude", 0.0)
    severity = complaint.get("severity", "Moderate").upper()
    address = complaint.get("address", f"Lat: {lat:.5f}, Lon: {lon:.5f}")
    time_str = complaint.get("reported_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    note = complaint.get("note", "Reported by commuter via RideVision AI")

    return (
        f"🚨 *POTHOLE HAZARD REPORT — RideVision*\n\n"
        f"📍 *Location:* {address}\n"
        f"🌐 *Google Maps:* https://maps.google.com/?q={lat},{lon}\n"
        f"⚠️ *Severity:* {severity}\n"
        f"🕒 *Reported At:* {time_str}\n"
        f"📝 *Note:* {note}\n\n"
        f"_This report was verified and auto-routed via the RideVision Computer Vision Commuter Safety System._"
    )


def route_complaint(complaint: Dict[str, Any], city_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates actionable routing details based on city authority channel:
    - WhatsApp: direct clickable link (wa.me) with prefilled complaint text.
    - Helpline: phone number and instructions.
    - Email: mailto link with subject and body.
    """
    channel = city_config.get("channel_type", "none")
    contact = city_config.get("contact_value", "")
    authority = city_config.get("authority_name", "Municipal Authority")
    message = build_complaint_text(complaint)

    result = {
        "city": city_config.get("city_name", "Unknown"),
        "authority": authority,
        "channel": channel,
        "instructions": city_config.get("instructions", ""),
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
        subject = quote(f"Road Pothole Hazard Report - {city_config.get('city_name', '')}")
        body = quote(message)
        result["link"] = f"mailto:{contact}?subject={subject}&body={body}"

    elif channel == "helpline":
        result["action"] = "call_helpline"
        result["link"] = f"tel:{contact}"

    return result
