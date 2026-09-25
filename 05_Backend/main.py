"""
RideVision FastAPI Server.
Core REST API for on-device/server pothole detection, spatial hazard warnings ahead,
trip route verification, and civic authority complaint routing.
"""

import os
import io
import base64
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from database import (
    init_db,
    get_all_potholes,
    get_pothole_by_id,
    save_or_merge_pothole,
    record_confirmation,
    get_last_user_confirmation,
    get_city_config,
    get_analytics_summary
)
from geo_matching import (
    Pothole,
    TripPoint,
    check_warning_ahead,
    match_trip_passed_by,
    is_confirmation_eligible,
    get_city_from_coords,
    route_complaint
)
from detector import detector

# Initialize directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RideVision API",
    description="CV-Based Pothole Detection, Real-time Hazard Warning, and Civic Grievance Routing",
    version="2.0.0"
)

# Enable CORS for mobile apps and frontend dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for uploaded and sample images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Ensure database tables & seed data are initialized immediately
init_db()
print("[RideVision API] Database initialized & ready.")


@app.on_event("startup")
def on_startup():
    init_db()


# --- Request/Response Models ---

class PotholeReportPayload(BaseModel):
    lat: float
    lon: float
    severity: str = "moderate"  # minor, moderate, severe
    note: Optional[str] = None
    address: Optional[str] = None
    user_id: str = "commuter-user"
    image_base64: Optional[str] = None


class CheckAheadPayload(BaseModel):
    lat: float
    lon: float
    heading: float = Field(..., ge=0, le=360, description="Driver compass heading 0-360")
    speed_kmh: Optional[float] = None
    alert_radius_m: float = 250.0
    heading_tolerance_deg: float = 45.0


class TripPointModel(BaseModel):
    lat: float
    lon: float
    heading: Optional[float] = None
    speed_kmh: Optional[float] = None
    timestamp: Optional[str] = None


class TripCompletePayload(BaseModel):
    trip_id: str
    user_id: str = "commuter-user"
    points: List[TripPointModel]
    proximity_radius_m: float = 25.0


class ConfirmationPayload(BaseModel):
    user_id: str
    confirmation_type: str = Field(..., pattern="^(still_there|fixed)$")


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "system": "RideVision API",
        "status": "online",
        "detector_engine": "YOLOv8" if detector.is_yolo_loaded else "OpenCV CV Fallback",
        "version": "2.0.0",
        "endpoints": [
            "/api/detect",
            "/api/potholes/report",
            "/api/potholes",
            "/api/trip/check-ahead",
            "/api/trip/complete",
            "/api/municipal/route/{id}",
            "/api/analytics/stats"
        ]
    }


@app.post("/api/detect")
async def detect_pothole_frame(
    file: Optional[UploadFile] = File(None),
    conf_threshold: float = Query(0.35, ge=0.1, le=0.9)
):
    """
    Runs computer vision inference on an uploaded frame.
    Returns detected bounding boxes, severity rating, and an annotated image.
    """
    if not file:
        raise HTTPException(status_code=400, detail="Image file must be provided.")

    contents = await file.read()
    try:
        detections, annotated_bgr = detector.detect(contents, conf_threshold=conf_threshold)

        # Encode annotated image to base64 JPEG
        import cv2
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        max_severity = "none"
        if detections:
            severities = [d["severity"] for d in detections]
            if "severe" in severities:
                max_severity = "severe"
            elif "moderate" in severities:
                max_severity = "moderate"
            else:
                max_severity = "minor"

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "overall_severity": max_severity,
            "annotated_image_base64": f"data:image/jpeg;base64,{annotated_b64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/potholes/report")
def report_pothole(payload: PotholeReportPayload):
    """
    Registers a pothole report.
    Automatically identifies the city, dedupes against nearby existing potholes (within 15m),
    and generates municipal dispatch links (e.g. WhatsApp for MCC Mangaluru, BBMP helpline).
    """
    city = get_city_from_coords(payload.lat, payload.lon)
    saved_img_rel_path = None

    # Handle image base64 if provided
    if payload.image_base64:
        try:
            raw_data = payload.image_base64
            if "," in raw_data:
                raw_data = raw_data.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_data)
            fname = f"pothole_{uuid.uuid4().hex[:10]}.jpg"
            fpath = os.path.join(UPLOADS_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            saved_img_rel_path = f"/static/uploads/{fname}"
        except Exception as e:
            print(f"Warning: Failed to save base64 image: {e}")

    # Persist and deduplicate
    result = save_or_merge_pothole(
        lat=payload.lat,
        lon=payload.lon,
        city=city,
        severity=payload.severity,
        user_id=payload.user_id,
        image_path=saved_img_rel_path,
        note=payload.note,
        address=payload.address
    )

    pothole = result["pothole"]
    city_config = get_city_config(city)
    routing = route_complaint(pothole, city_config)

    return {
        "success": True,
        "is_new_hazard": result["is_new"],
        "pothole": pothole,
        "report_id": result["report_id"],
        "municipal_routing": routing
    }


@app.get("/api/potholes")
def list_potholes(
    status: Optional[str] = Query(None, description="active | reported_fixed | verified_fixed"),
    city: Optional[str] = Query(None, description="City name filter e.g. Mangaluru")
):
    """Lists potholes registered in the system with optional city and status filters."""
    potholes = get_all_potholes(status=status, city=city)
    return {
        "count": len(potholes),
        "potholes": potholes
    }


@app.get("/api/potholes/{pothole_id}")
def get_pothole(pothole_id: str):
    """Fetches details for a single pothole including current routing links."""
    pothole = get_pothole_by_id(pothole_id)
    if not pothole:
        raise HTTPException(status_code=404, detail="Pothole not found")

    city_config = get_city_config(pothole["city"])
    routing = route_complaint(pothole, city_config)

    return {
        "pothole": pothole,
        "municipal_routing": routing
    }


@app.post("/api/trip/check-ahead")
def check_ahead(payload: CheckAheadPayload):
    """
    Called by commuter's device periodically while driving (e.g. every 2-3 seconds).
    Detects if an active pothole is ahead in the driving cone and returns an immediate warning.
    """
    db_potholes = get_all_potholes(status="active")
    pothole_objects = [
        Pothole(
            id=p["id"],
            lat=p["lat"],
            lon=p["lon"],
            city=p["city"],
            severity=p["severity"],
            status=p["status"],
            confirmation_count=p["confirmation_count"],
            address=p.get("address")
        )
        for p in db_potholes
    ]

    warning = check_warning_ahead(
        current_lat=payload.lat,
        current_lon=payload.lon,
        heading_deg=payload.heading,
        all_potholes=pothole_objects,
        alert_radius_m=payload.alert_radius_m,
        heading_tolerance_deg=payload.heading_tolerance_deg,
        min_confirmation_count=1
    )

    return {
        "has_warning": warning is not None,
        "warning": warning
    }


@app.post("/api/trip/complete")
def complete_trip(payload: TripCompletePayload):
    """
    Analyzes breadcrumbs of a completed commute route and returns potholes passed by,
    generating "Is this pothole still there?" verification cards for the driver.
    """
    db_potholes = get_all_potholes()
    pothole_objects = [
        Pothole(
            id=p["id"],
            lat=p["lat"],
            lon=p["lon"],
            city=p["city"],
            severity=p["severity"],
            status=p["status"],
            confirmation_count=p["confirmation_count"],
            address=p.get("address")
        )
        for p in db_potholes
    ]

    trip_points = [
        TripPoint(
            lat=pt.lat,
            lon=pt.lon,
            heading=pt.heading,
            speed_kmh=pt.speed_kmh,
            timestamp=pt.timestamp
        )
        for pt in payload.points
    ]

    passed = match_trip_passed_by(trip_points, pothole_objects, payload.proximity_radius_m)

    return {
        "trip_id": payload.trip_id,
        "potholes_passed_count": len(passed),
        "passed_potholes": passed
    }


@app.post("/api/potholes/{pothole_id}/confirm")
def confirm_pothole(pothole_id: str, payload: ConfirmationPayload):
    """
    Submits a verification confirmation ('still_there' or 'fixed').
    Enforces a 7-day cooldown per user per pothole to prevent manipulation.
    """
    pothole = get_pothole_by_id(pothole_id)
    if not pothole:
        raise HTTPException(status_code=404, detail="Pothole not found")

    last_conf = get_last_user_confirmation(payload.user_id, pothole_id)
    if not is_confirmation_eligible(payload.user_id, pothole_id, last_conf, cooldown_days=7):
        raise HTTPException(
            status_code=429,
            detail="Verification Cooldown Active: You have already submitted a confirmation for this hazard within the last 7 days."
        )

    res = record_confirmation(payload.user_id, pothole_id, payload.confirmation_type)
    return {
        "success": True,
        "message": f"Successfully recorded '{payload.confirmation_type}' confirmation.",
        "pothole": res["pothole"]
    }


@app.get("/api/municipal/route/{pothole_id}")
def get_complaint_route(pothole_id: str):
    """Returns municipal dispatch instructions and links for a given pothole."""
    pothole = get_pothole_by_id(pothole_id)
    if not pothole:
        raise HTTPException(status_code=404, detail="Pothole not found")

    city_config = get_city_config(pothole["city"])
    routing = route_complaint(pothole, city_config)
    return routing


@app.get("/api/analytics/stats")
def get_stats():
    """Returns aggregated road safety analytics and municipal resolution statistics."""
    return get_analytics_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
