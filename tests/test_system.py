"""
Comprehensive End-to-End System Tests for RideVision.
Validates Model Inference, Geo-Matching, Database Deduplication,
Anti-Gaming Cooldown, and FastAPI REST Endpoints.
"""

import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from PIL import Image

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "05_Backend")
sys.path.insert(0, BACKEND_DIR)

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
    haversine_distance_m,
    bearing_deg,
    angle_diff_deg,
    check_warning_ahead,
    match_trip_passed_by,
    is_confirmation_eligible,
    get_city_from_coords,
    route_complaint
)
from detector import detector
import main
from fastapi.testclient import TestClient


class TestRideVisionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(main.app)

    def test_01_geodesic_math(self):
        """Test Haversine distance, compass bearing, and angular difference calculations."""
        # Distance between SJEC Vamanjoor (12.9152, 74.8988) and Kankanady (12.8715, 74.8564) ~ 6.6 km
        dist = haversine_distance_m(12.9152, 74.8988, 12.8715, 74.8564)
        self.assertTrue(6000 < dist < 7200, f"Expected ~6.6km, got {dist}m")

        # Bearing calculation
        bearing = bearing_deg(12.9152, 74.8988, 12.8715, 74.8564)
        self.assertTrue(200 < bearing < 260, f"Expected SW bearing, got {bearing}")

        # Angular difference
        diff1 = angle_diff_deg(10, 350)
        self.assertEqual(diff1, 20.0)
        diff2 = angle_diff_deg(180, 200)
        self.assertEqual(diff2, 20.0)

    def test_02_warning_ahead_logic(self):
        """Test the 250m and +/-45 deg directional cone warning system."""
        potholes = [
            Pothole(
                id="test-p1", lat=12.8715, lon=74.8564, city="Mangaluru",
                severity="severe", status="active", confirmation_count=5, address="Kankanady Road"
            )
        ]

        # Scenario A: Vehicle approaching 140m away, heading 245 deg (directly aligned) -> Warning should fire
        warning = check_warning_ahead(
            current_lat=12.8724, current_lon=74.8573, heading_deg=245.0,
            all_potholes=potholes, alert_radius_m=250.0, heading_tolerance_deg=45.0
        )
        self.assertIsNotNone(warning, "Warning should be triggered when hazard is ahead inside alert cone")
        self.assertEqual(warning["pothole_id"], "test-p1")
        self.assertTrue(warning["distance_m"] < 250)

        # Scenario B: Vehicle driving in opposite direction (Heading 65 deg NE, moving away) -> Warning should NOT fire
        warning_away = check_warning_ahead(
            current_lat=12.8724, current_lon=74.8573, heading_deg=65.0,
            all_potholes=potholes, alert_radius_m=250.0, heading_tolerance_deg=45.0
        )
        self.assertIsNone(warning_away, "Warning should not fire when vehicle is driving away from pothole")

    def test_03_city_detection_and_routing(self):
        """Test jurisdiction detection and WhatsApp link generation."""
        # Mangaluru coordinates
        city = get_city_from_coords(12.9152, 74.8988)
        self.assertEqual(city, "Mangaluru")

        # Routing config lookup
        config = get_city_config("Mangaluru")
        self.assertEqual(config["channel_type"], "whatsapp")
        self.assertIn("919449007722", config["contact_value"])

        # WhatsApp link generation
        complaint = {
            "lat": 12.9152, "lon": 74.8988, "severity": "severe",
            "address": "NH 73 near SJEC", "reported_at": "2026-09-14 18:00"
        }
        routed = route_complaint(complaint, config)
        self.assertEqual(routed["channel"], "whatsapp")
        self.assertIn("wa.me/919449007722", routed["link"])
        self.assertIn("POTHOLE%20HAZARD%20REPORT", routed["link"])

    def test_04_anti_gaming_cooldown(self):
        """Test that the 7-day cooldown prevents confirmation spam."""
        user_id = "test-commuter-spammer"
        pothole_id = "pothole-mng-001"

        # Case 1: First time confirming -> Eligible
        self.assertTrue(is_confirmation_eligible(user_id, pothole_id, None, cooldown_days=7))

        # Case 2: Confirmed 1 hour ago -> Ineligible (blocked)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        self.assertFalse(is_confirmation_eligible(user_id, pothole_id, recent_time, cooldown_days=7))

        # Case 3: Confirmed 8 days ago -> Eligible again
        old_time = datetime.now(timezone.utc) - timedelta(days=8)
        self.assertTrue(is_confirmation_eligible(user_id, pothole_id, old_time, cooldown_days=7))

    def test_05_pothole_deduplication(self):
        """Test spatial deduplication: reports within 15m merge into one master pothole."""
        base_lat, base_lon = 12.91520, 74.89880

        # Report 1
        rep1 = save_or_merge_pothole(
            lat=base_lat, lon=base_lon, city="Mangaluru", severity="severe",
            user_id="user-a", address="SJEC Road Spot"
        )
        pothole_id_1 = rep1["pothole"]["id"]

        # Report 2: only 5 meters away
        rep2 = save_or_merge_pothole(
            lat=base_lat + 0.00004, lon=base_lon + 0.00003, city="Mangaluru", severity="severe",
            user_id="user-b", address="SJEC Road Spot"
        )
        pothole_id_2 = rep2["pothole"]["id"]

        # Should merge into the same pothole ID
        self.assertEqual(pothole_id_1, pothole_id_2, "Reports within 15m must merge into existing hazard")
        self.assertFalse(rep2["is_new"], "Second close report should not be flagged as new hazard")

    def test_06_fastapi_rest_endpoints(self):
        """Test FastAPI endpoints via TestClient."""
        # 1. Health check
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

        # 2. List potholes
        res_list = self.client.get("/api/potholes?city=Mangaluru")
        self.assertEqual(res_list.status_code, 200)
        self.assertTrue(res_list.json()["count"] > 0)

        # 3. Check warning ahead
        res_warn = self.client.post("/api/trip/check-ahead", json={
            "lat": 12.8724, "lon": 74.8573, "heading": 245.0
        })
        self.assertEqual(res_warn.status_code, 200)
        self.assertTrue(res_warn.json()["has_warning"])

        # 4. Analytics stats
        res_stats = self.client.get("/api/analytics/stats")
        self.assertEqual(res_stats.status_code, 200)
        self.assertIn("total_potholes", res_stats.json())

    def test_07_cv_detector_inference(self):
        """Test that the Computer Vision detector processes frames and returns structured bounding boxes."""
        sample_path = os.path.join(PROJECT_ROOT, "06_Web_Dashboard", "samples", "sample_severe_pothole.jpg")
        self.assertTrue(os.path.exists(sample_path))

        img = Image.open(sample_path)
        detections, annotated = detector.detect(img, conf_threshold=0.30, engine="hybrid")
        self.assertTrue(len(detections) > 0, "Detector should detect the pothole cavity")
        self.assertIn("severity", detections[0])
        self.assertEqual(detections[0]["severity"], "severe")
        self.assertIsNotNone(annotated)


if __name__ == "__main__":
    unittest.main()
