"""
RideVision — Developer & Engineering Portal.
St Joseph Engineering College (SJEC) – Dept. of AIML, PBL Project.

Unified Technical Console for:
  1. 🧠 CV Model & Inference Benchmarks (YOLOv8 vs OpenCV, latency, FPS, tensor metrics)
  2. 🚗 Drive & Hazard HUD Simulation (OSRM routing, 250m directional cone hazard warning)
  3. 🗺️ Live Road Safety Map (Spatial hazard visualization, city & status filters)
  4. 🏛️ Database & Civic Routing Admin (SQLite inspection, lifecycle status override, seed reset)
  5. ⚡ REST API Telemetry & Tester (FastAPI backend health check and client testing)
  6. 🎓 Academic PBL Viva Defense & Specs (Formulations, architecture comparison, defense Q&A)
"""

import os
import sys
import time
import json
import glob
import urllib.request
import urllib.error
from datetime import datetime, timezone
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import pandas as pd
import pydeck as pdk

DATASET_IMAGES_DIR = r"C:\Users\saich\Desktop\datasets\train\images"

def get_dataset_samples(limit=25):
    """Returns a dict of {display_name: full_path} for real dataset images."""
    if not os.path.isdir(DATASET_IMAGES_DIR):
        return {}
    files = sorted(glob.glob(os.path.join(DATASET_IMAGES_DIR, "*.png")) + glob.glob(os.path.join(DATASET_IMAGES_DIR, "*.jpg")))[:limit]
    return {f"📁 Dataset: {os.path.basename(f)}": f for f in files}

# Add 05_Backend to Python search path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "05_Backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import importlib
import database as database_mod
import geo_matching as geo_matching_mod
import detector as detector_mod

try:
    importlib.reload(database_mod)
    importlib.reload(geo_matching_mod)
    importlib.reload(detector_mod)
except Exception:
    pass

from database import (
    init_db,
    get_all_potholes,
    get_pothole_by_id,
    save_or_merge_pothole,
    record_confirmation,
    get_last_user_confirmation,
    get_city_config,
    get_analytics_summary,
    update_pothole_status,
    get_all_reports,
    add_or_update_city_config,
    reset_database
)
from geo_matching import (
    Pothole,
    TripPoint,
    haversine_distance_m,
    bearing_deg,
    check_warning_ahead,
    match_trip_passed_by,
    is_confirmation_eligible,
    get_city_from_coords,
    reverse_geocode,
    geocode_address,
    detect_ip_location,
    get_route_osrm,
    parse_gpx_content,
    parse_csv_gps_log,
    parse_gps_track,
    load_gpx_file,
    route_complaint
)
detector = detector_mod.detector

# Ensure database is initialized
init_db()

# Page configuration
st.set_page_config(
    page_title="RideVision | Developer Engineering Console",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PREMIUM MODERN CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1300px;
    }

    /* Header Titles */
    .hero-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, rgba(26, 34, 56, 0.75) 0%, rgba(15, 20, 32, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Mode Pill Badge */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-dev {
        background: rgba(0, 210, 255, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(0, 210, 255, 0.35);
    }

    /* Cards */
    .glass-card {
        background: rgba(18, 26, 44, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* Driver HUD Warning Elements */
    .hud-warning {
        background: linear-gradient(135deg, rgba(185, 28, 28, 0.35) 0%, rgba(127, 29, 29, 0.45) 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.4rem;
        margin: 1rem 0;
        animation: hudPulse 1.8s infinite alternate;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    }
    @keyframes hudPulse {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); border-color: #EF4444; }
        100% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.5); border-color: #F87171; }
    }
    .hud-safe {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(4, 47, 36, 0.45) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.4rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .hud-title-danger {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F87171;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hud-title-safe {
        font-size: 1.4rem;
        font-weight: 800;
        color: #34D399;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Monospace tags */
    .code-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background: rgba(0, 0, 0, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: SYSTEM METRICS & VISION ENGINE TELEMETRY
# ==============================================================================
st.sidebar.markdown("### 🛠️ **RideVision Console**")
st.sidebar.caption("Developer, Research & Engineering Portal")
st.sidebar.markdown("---")

st.sidebar.markdown("#### 📊 **Live Safety Metrics**")
stats = get_analytics_summary()
c_s1, c_s2 = st.sidebar.columns(2)
c_s1.metric("Active Hazards", stats["active_potholes"])
c_s2.metric("Verified Fixed", stats["verified_fixed"])
c_s3, c_s4 = st.sidebar.columns(2)
c_s3.metric("Confirmations", stats["total_confirmations"])
c_s4.metric("Civic Fix Rate", f"{stats['fix_rate_percent']}%")

st.sidebar.markdown("---")
detector_status = "🟢 YOLOv8 Nano" if detector.is_yolo_loaded else "🔴 Model Initializing"
st.sidebar.caption(f"**Vision Engine:** {detector_status}")
st.sidebar.caption("**Backend API:** `FastAPI @ :8000`")
st.sidebar.caption("St Joseph Engineering College (SJEC) – AIML")

# ==============================================================================
# APP HEADER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div>
        <div class="hero-title">RideVision Engineering Console 🛠️ <span class="mode-badge badge-dev">Developer Portal</span></div>
        <div class="hero-subtitle">YOLOv8 Edge Benchmarks • Live Drive & HUD Simulation • Spatial Safety Map • Database Admin • REST Telemetry • PBL Viva Defense</div>
    </div>
    <div style="text-align: right;">
        <span class="code-tag" style="color: #38BDF8;">FastAPI :8000</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# UNIFIED DEVELOPER & ENGINEERING CONSOLE TABS
# ==============================================================================
tab_model, tab_sim, tab_map, tab_db, tab_api, tab_pbl = st.tabs([
    "🧠 CV Model & Inference Benchmarks",
    "🚗 Drive & HUD Simulation",
    "🗺️ Road Safety Map",
    "🏛️ Database & Civic Routing Admin",
    "⚡ REST API Telemetry & Tester",
    "🎓 PBL Viva Defense & Academic Specs"
])

# ------------------------------------------------------------------------------
# TAB 1: CV MODEL & INFERENCE LAB
# ------------------------------------------------------------------------------
with tab_model:
    st.markdown("### 🧠 YOLOv8 Computer Vision & Edge Model Benchmarks")
    st.write("Inspect YOLOv8 neural network inference, adjust confidence cutoff and NMS IoU suppression hyper-parameters, and inspect bounding box tensor metrics.")

    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    col_dev_img, col_dev_res = st.columns([1, 1.4])

    def find_sample_file(base_name):
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(samples_dir, base_name + ext)
            if os.path.exists(p):
                return p
        return os.path.join(samples_dir, base_name + ".jpg")

    with col_dev_img:
        st.markdown("#### 1. Input Test Frame")
        bench_picks = {}
        s1 = find_sample_file("sample_severe_pothole")
        if os.path.exists(s1):
            bench_picks["Sample 1: Severe Road Cavity"] = s1
        s2 = find_sample_file("sample_moderate_pothole")
        if os.path.exists(s2):
            bench_picks["Sample 2: Moderate Asphalt Cavity"] = s2
        s3 = find_sample_file("sample_clean_road")
        if os.path.exists(s3):
            bench_picks["Sample 3: Clean Road Control"] = s3

        # Include real dataset images if available
        bench_picks.update(get_dataset_samples(limit=25))

        test_img_pick = st.selectbox("Select Benchmark Frame:", list(bench_picks.keys()))
        loaded_img = Image.open(bench_picks[test_img_pick])
        st.image(loaded_img, caption="Benchmark Input Frame (640x640 normalized)", use_container_width=True)

        st.markdown("#### 2. YOLOv8 Inference Hyperparameters")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            dev_conf = st.slider("Confidence Cutoff Threshold:", 0.05, 0.95, 0.40, 0.05, help="Minimum confidence required to classify a detected cavity")
        with c_p2:
            dev_iou = st.slider("NMS IoU Suppression Threshold:", 0.10, 0.80, 0.35, 0.05, help="Controls overlap suppression to eliminate duplicate boxes on same pothole")

    with col_dev_res:
        st.markdown("#### 3. Execution Telemetry")
        t_start = time.perf_counter()
        detections, ann_bgr = detector.detect(loaded_img, conf_threshold=dev_conf, iou_threshold=dev_iou)
        t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        ann_rgb = Image.fromarray(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)) if 'cv2' in sys.modules else loaded_img

        # Benchmarks
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Latency</div><div class="metric-value">{t_elapsed_ms} <span style="font-size:0.9rem;color:#38BDF8;">ms</span></div></div>', unsafe_allow_html=True)
        with b2:
            fps = round(1000.0 / t_elapsed_ms, 1) if t_elapsed_ms > 0 else 0
            st.markdown(f'<div class="glass-card"><div class="metric-label">Throughput</div><div class="metric-value">{fps} <span style="font-size:0.9rem;color:#38BDF8;">FPS</span></div></div>', unsafe_allow_html=True)
        with b3:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Detections</div><div class="metric-value">{len(detections)} <span style="font-size:0.9rem;color:#38BDF8;">boxes</span></div></div>', unsafe_allow_html=True)

        st.image(ann_rgb, caption="Annotated Bounding Box Output", use_container_width=True)

        if detections:
            st.markdown("#### 4. Bounding Box Tensor Metrics")
            df_boxes = pd.DataFrame([
                {
                    "Box ID": f"DET-{i+1}",
                    "Confidence": f"{round(d['confidence']*100, 2)}%",
                    "Severity": d["severity"].upper(),
                    "Box [x1, y1, x2, y2]": str(d["box"]),
                    "Surface Area %": f"{round(d.get('area_ratio', 0)*100, 2)}%"
                }
                for i, d in enumerate(detections)
            ])
            st.dataframe(df_boxes, use_container_width=True)

        st.markdown("#### 5. Model Edge Deployment Specs")
        st.markdown("""
        - **Model Architecture:** YOLOv8 nano (`yolov8n.pt`) single-class pothole detector.
        - **Parameter Count:** ~3.2 Million parameters.
        - **Weight Artifacts:** PyTorch weights: `best.pt` (6.2 MB), TFLite edge model: `best.tflite` (12.2 MB).
        - **Mobile Integration:** Bundled in `03_Mobile_Android/app/src/main/assets/best.tflite` with CameraX inference pipeline.
        """)

# ------------------------------------------------------------------------------
# TAB 2: DRIVE & HUD SIMULATION
# ------------------------------------------------------------------------------
with tab_sim:
    st.markdown("### 🚗 Live Driver Head-Up Display (HUD) & Route Simulation")
    st.write("Real-time safety navigation workbench: Simulates vehicle movement along real road corridors, continuously evaluating active potholes within a **250m forward directional cone** (±45° heading tolerance).")

    col_sim_ctrl, col_hud_display = st.columns([1.1, 1.4])

    with col_sim_ctrl:
        st.markdown("#### 🛣️ Recorded Commute GPS Route")
        csv_path = os.path.join(os.path.dirname(__file__), "commute_route.csv")
        gpx_path = os.path.join(os.path.dirname(__file__), "gpx_routes", "mangaluru_user_recorded_commute.gpx")
        root_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "commute_route.csv")
        
        if os.path.exists(csv_path):
            route_data = load_gpx_file(csv_path)
        elif os.path.exists(root_csv_path):
            route_data = load_gpx_file(root_csv_path)
        elif os.path.exists(gpx_path):
            route_data = load_gpx_file(gpx_path)
        else:
            # Fallback inline coordinates from recorded log
            route_data = {
                "name": "Mangaluru: Kudupu to SJEC Vamanjoor Commute Log",
                "coordinates": [
                    [12.9212923, 74.8651359],
                    [12.9187606, 74.8680822],
                    [12.9160795, 74.8755951],
                    [12.9176948, 74.8804785],
                    [12.9152294, 74.8839134],
                    [12.9150299, 74.8874427],
                    [12.9146838, 74.8875795],
                    [12.9142577, 74.8886940],
                    [12.9141977, 74.8884412],
                    [12.9120148, 74.8917148],
                    [12.9104978, 74.8965336],
                    [12.9117471, 74.8990154],
                    [12.9121512, 74.9002380],
                    [12.9122396, 74.8998308]
                ],
                "distance_km": 4.36,
                "duration_min": 7.5,
                "total_points": 14,
                "source": "gps_csv_log"
            }

        route_coords = route_data["coordinates"]
        total_pts = len(route_coords)
        track_title = route_data.get("name", "Hardwired Commute GPS Route")
        start_name = f"Start ({route_coords[0][0]:.5f}, {route_coords[0][1]:.5f})" if total_pts > 0 else "Start"
        end_name = f"Destination ({route_coords[-1][0]:.5f}, {route_coords[-1][1]:.5f})" if total_pts > 0 else "Destination"

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.25); margin-bottom: 12px;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem; margin-bottom: 6px;">
                📍 {track_title}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: #E2E8F0;">
                <span>🚩 <b>Start:</b> {start_name}</span>
                <span>🏁 <b>Dest:</b> {end_name}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: #94A3B8; margin-top: 8px;">
                <span>🛣️ Distance: <b style="color:#FFFFFF;">{route_data['distance_km']} km</b></span>
                <span>⏱️ Est. Drive Time: <b style="color:#FFFFFF;">{route_data['duration_min']} mins</b></span>
                <span>📍 GPS Waypoints: <b style="color:#FFFFFF;">{total_pts} Logged Points</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Check all active potholes in DB
        all_active = get_all_potholes(status="active")
        p_objects = [
            Pothole(
                id=p["id"], lat=p["lat"], lon=p["lon"], city=p["city"],
                severity=p["severity"], status=p["status"], confirmation_count=p["confirmation_count"],
                address=p.get("address")
            )
            for p in all_active
        ]

        # Find potholes along this commute path (within 300m corridor)
        route_potholes = []
        for p in p_objects:
            for wpt in route_coords[::max(1, total_pts // 50)]:
                if haversine_distance_m(wpt[0], wpt[1], p.lat, p.lon) <= 300:
                    route_potholes.append(p)
                    break

        if route_potholes:
            st.warning(f"⚠️ **Hazard Notice:** Found {len(route_potholes)} active pothole(s) along this commute road corridor.")
        else:
            st.success("✅ **Smooth Corridor:** No severe road cavities currently recorded along this direct path.")

        # Drive simulation slider
        sim_progress = st.slider(
            "Simulate Driving Along Route (%):",
            min_value=0, max_value=100, value=25, step=1,
            help="Slide to simulate moving vehicle position along the real road network"
        )

        # Current vehicle index and coordinates
        curr_idx = int((sim_progress / 100.0) * (total_pts - 1))
        curr_pt = route_coords[curr_idx]
        curr_lat, curr_lon = curr_pt[0], curr_pt[1]

        # Compute dynamic compass heading towards next point
        if curr_idx < total_pts - 1:
            next_pt = route_coords[curr_idx + 1]
            calc_heading = bearing_deg(curr_lat, curr_lon, next_pt[0], next_pt[1])
        elif curr_idx > 0:
            prev_pt = route_coords[curr_idx - 1]
            calc_heading = bearing_deg(prev_pt[0], prev_pt[1], curr_lat, curr_lon)
        else:
            calc_heading = 180.0

    with col_hud_display:
        warning = check_warning_ahead(
            current_lat=curr_lat,
            current_lon=curr_lon,
            heading_deg=calc_heading,
            all_potholes=p_objects,
            alert_radius_m=250.0,
            heading_tolerance_deg=45.0,
            min_confirmation_count=1
        )

        # Telemetry Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            sim_speed = 45 if sim_progress < 95 else 15
            st.markdown(f'<div class="glass-card"><div class="metric-label">Vehicle Speed</div><div class="metric-value">{sim_speed} <span style="font-size:0.9rem;color:#94A3B8;">km/h</span></div></div>', unsafe_allow_html=True)
        with m_col2:
            dist_display = f"{warning['distance_m']} m" if warning else "Clear"
            st.markdown(f'<div class="glass-card"><div class="metric-label">Hazard Distance</div><div class="metric-value">{dist_display}</div></div>', unsafe_allow_html=True)
        with m_col3:
            cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            cardinal = cardinals[int(((calc_heading + 22.5) % 360) / 45)]
            st.markdown(f'<div class="glass-card"><div class="metric-label">Compass Heading</div><div class="metric-value">{int(calc_heading)}° <span style="font-size:0.9rem;color:#94A3B8;">{cardinal}</span></div></div>', unsafe_allow_html=True)

        if warning:
            is_imminent = warning["distance_m"] <= 80
            badge_text = "🚨 IMMINENT COLLISION RISK" if is_imminent else "⚠️ PROACTIVE HAZARD WARNING"
            advice = "Brake gently and steer clear of lane cavity." if is_imminent else "Reduce speed to 25 km/h. Maintain safe following distance."

            st.markdown(f"""
            <div class="hud-warning">
                <div class="hud-title-danger">
                    {badge_text}
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-top: 6px;">
                    {warning['severity'].upper()} POTHOLE DETECTED {warning['distance_m']} METERS AHEAD
                </div>
                <div style="color: #FCA5A5; font-size: 0.95rem; margin-top: 8px; line-height: 1.5;">
                    📍 <b>Location:</b> {warning['address']}<br>
                    📐 <b>Alignment:</b> Direct forward corridor ({warning['angular_deviation_deg']}° off heading)<br>
                    👥 <b>Community Confirmations:</b> {warning['confirmation_count']} verification vote(s)
                </div>
                <div style="margin-top: 12px; padding: 8px 14px; background: rgba(0, 0, 0, 0.35); border-radius: 8px; border-left: 4px solid #EF4444; font-weight: 600; color: #FECACA;">
                    🛡️ <b>Driver Action:</b> {advice}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hud-safe">
                <div class="hud-title-safe">
                    ✅ ROAD AHEAD IS SAFE & CLEAR
                </div>
                <div style="font-size: 1.05rem; color: #D1FAE5; margin-top: 6px;">
                    No severe road hazards detected within your 250m forward driving corridor.<br>
                    Vehicle progressing smoothly at {sim_progress}% of commute.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # PyDeck 3D Map with Route Path + Vehicle Marker + Potholes
        path_df = pd.DataFrame([{
            "path": [[pt[1], pt[0]] for pt in route_coords],
            "color": [14, 165, 233, 200]
        }])

        # Markers: Vehicle + Potholes
        marker_list = [
            {"lat": curr_lat, "lon": curr_lon, "name": "Simulated Vehicle", "color": [0, 210, 255, 255], "radius": 40}
        ]
        for p in p_objects:
            p_color = [239, 68, 68, 230] if p.severity == "severe" else [245, 158, 11, 230]
            marker_list.append({
                "lat": p.lat, "lon": p.lon,
                "name": f"{p.severity.upper()} Pothole: {p.address or 'Hazard'}",
                "color": p_color, "radius": 25
            })

        map_df = pd.DataFrame(marker_list)

        st.pydeck_chart(pdk.Deck(
            map_style="road",
            initial_view_state=pdk.ViewState(latitude=curr_lat, longitude=curr_lon, zoom=14, pitch=35),
            layers=[
                pdk.Layer(
                    "PathLayer",
                    path_df,
                    get_path="path",
                    get_color="color",
                    width_min_pixels=4,
                    pickable=False
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    map_df,
                    get_position="[lon, lat]",
                    get_color="color",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True
                )
            ],
            tooltip={"html": "<b>{name}</b>", "style": {"backgroundColor": "#0F172A", "color": "#FFFFFF", "padding": "6px", "borderRadius": "6px"}}
        ))

# ------------------------------------------------------------------------------
# TAB 3: ROAD SAFETY MAP
# ------------------------------------------------------------------------------
with tab_map:
    st.markdown("### 🗺️ Live Road Safety Map")
    st.write("Interactive spatial GIS layer of recorded road hazards, severity distributions, and verified repaired stretches across monitored municipal jurisdictions.")

    # Dynamic city list from database
    all_cities_in_db = [c["city"] for c in get_all_potholes() if c.get("city")]
    unique_cities = sorted(list(set(all_cities_in_db + ["Mangaluru", "Bengaluru", "Udupi", "Mysuru"])))
    city_options = ["All Cities"] + unique_cities

    map_c1, map_c2 = st.columns([1, 1])
    with map_c1:
        u_map_city = st.selectbox("Select City Jurisdiction:", city_options, index=0)
    with map_c2:
        u_map_status = st.selectbox("Status Filter:", ["All", "Active Hazards", "Verified Fixed"])

    s_arg = None if u_map_status == "All" else ("active" if u_map_status == "Active Hazards" else "verified_fixed")
    c_arg = None if u_map_city == "All Cities" else u_map_city

    all_p_map = get_all_potholes(status=s_arg, city=c_arg)

    if all_p_map:
        p_records = []
        for p in all_p_map:
            if p["status"] == "verified_fixed":
                color = [16, 185, 129, 220]  # Emerald Green
                label_status = "REPAIRED"
                radius = 55
            elif p["severity"] == "severe":
                color = [239, 68, 68, 230]   # Red
                label_status = "ACTIVE SEVERE"
                radius = 65
            else:
                color = [245, 158, 11, 230]  # Amber
                label_status = "ACTIVE MODERATE"
                radius = 55

            p_records.append({
                "id": p["id"],
                "lat": p["lat"],
                "lon": p["lon"],
                "city": p["city"],
                "address": p.get("address") or "Road Coordinate",
                "severity": p["severity"].upper(),
                "status": label_status,
                "confirmations": p["confirmation_count"],
                "color": color,
                "radius": radius,
                "maps_link": f"https://maps.google.com/?q={p['lat']},{p['lon']}"
            })

        df_p_map = pd.DataFrame(p_records)

        view_lat = df_p_map["lat"].mean()
        view_lon = df_p_map["lon"].mean()
        view_zoom = 12

        view_deck = pdk.Deck(
            map_style="road",
            initial_view_state=pdk.ViewState(
                latitude=view_lat,
                longitude=view_lon,
                zoom=view_zoom,
                pitch=35
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    df_p_map,
                    get_position="[lon, lat]",
                    get_color="color",
                    get_radius="radius",
                    pickable=True,
                    auto_highlight=True
                )
            ],
            tooltip={
                "html": "<b>{address}</b><br>City: {city}<br>Condition: <b>{status}</b><br>Confirmations: {confirmations}<br><a href='{maps_link}' target='_blank' style='color:#38BDF8;'>Open in Google Maps ↗</a>",
                "style": {"backgroundColor": "#0F172A", "color": "#FFFFFF", "borderRadius": "8px", "padding": "8px"}
            }
        )
        st.pydeck_chart(view_deck)

        # Quick Card Summary
        st.markdown("#### 📍 Hazard Stretches Table")
        st.dataframe(
            df_p_map[["id", "city", "address", "severity", "status", "confirmations"]],
            use_container_width=True
        )
    else:
        st.info("No road hazards found matching the selected filter.")

# ------------------------------------------------------------------------------
# TAB 4: DATABASE & CIVIC ROUTING ADMIN
# ------------------------------------------------------------------------------
with tab_db:
    st.markdown("### 🏛️ Database & Civic Routing Management")
    st.write("Inspect SQLite persistence layer (`ridevision.db`), update hazard lifecycle statuses, and manage city grievance dispatch channels.")

    db_view_tab1, db_view_tab2, db_view_tab3 = st.tabs([
        "📋 Potholes Master Table",
        "📥 User Reports Audit",
        "⚙️ City Routing Config & Reset"
    ])

    with db_view_tab1:
        raw_potholes = get_all_potholes()
        if raw_potholes:
            df_raw_p = pd.DataFrame(raw_potholes)
            st.dataframe(df_raw_p, use_container_width=True)

            st.markdown("#### ✏️ Manual Status Override")
            col_ov1, col_ov2, col_ov3 = st.columns([1.5, 1, 1])
            with col_ov1:
                p_to_edit = st.selectbox("Select Pothole ID:", [p["id"] for p in raw_potholes])
            with col_ov2:
                new_st = st.selectbox("New Lifecycle Status:", ["active", "reported_fixed", "verified_fixed"])
            with col_ov3:
                st.write("")
                st.write("")
                if st.button("Apply Status Update", type="primary"):
                    if update_pothole_status(p_to_edit, new_st):
                        st.success(f"Status of `{p_to_edit}` changed to `{new_st}`.")
                        time.sleep(1)
                        st.rerun()

    with db_view_tab2:
        reports_list = get_all_reports(limit=50)
        if reports_list:
            st.dataframe(pd.DataFrame(reports_list), use_container_width=True)
        else:
            st.info("No individual reports recorded yet.")

    with db_view_tab3:
        st.markdown("#### Municipal Forwarding Configurations")
        cities = ["Mangaluru", "Bengaluru", "Udupi", "Mysuru"]
        for c in cities:
            cfg = get_city_config(c)
            st.markdown(f"- **{c}** ({cfg['authority_name']}): Channel `{cfg['channel_type'].upper()}` -> `{cfg['contact_value']}`")

        st.markdown("---")
        st.markdown("#### ⚠️ Factory Database Reset")
        st.caption("Resets all tables and re-seeds initial sample hazards in Mangaluru (SJEC, Kankanady, Kadri) and Bengaluru (Indiranagar, Silk Board).")
        if st.button("🔄 Reset Database to Initial Seed State", type="secondary"):
            reset_database()
            st.success("Database has been reset and seeded successfully!")
            time.sleep(1)
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 5: REST API TELEMETRY & TESTER
# ------------------------------------------------------------------------------
with tab_api:
    st.markdown("### ⚡ Live FastAPI Telemetry & REST Client")
    st.write("Inspect the live FastAPI backend server (`http://localhost:8000`) and test REST API endpoints interactively.")

    api_url = "http://localhost:8000"
    
    # Check backend health
    backend_online = False
    backend_data = {}
    try:
        req = urllib.request.Request(f"{api_url}/", headers={"User-Agent": "RideVision-Dashboard"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.getcode() == 200:
                backend_online = True
                backend_data = json.loads(resp.read().decode())
    except Exception as e:
        backend_online = False

    c_api1, c_api2, c_api3 = st.columns(3)
    with c_api1:
        st.markdown(f'<div class="glass-card"><div class="metric-label">FastAPI Status</div><div class="metric-value" style="color:{"#10B981" if backend_online else "#EF4444"};">{"ONLINE" if backend_online else "OFFLINE"}</div></div>', unsafe_allow_html=True)
    with c_api2:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Version</div><div class="metric-value">{backend_data.get("version", "2.0.0")}</div></div>', unsafe_allow_html=True)
    with c_api3:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Swagger Docs</div><div class="metric-value"><a href="{api_url}/docs" target="_blank" style="color:#38BDF8;text-decoration:none;font-size:1.2rem;">Open UI ↗</a></div></div>', unsafe_allow_html=True)

    st.markdown("#### 🧪 Interactive REST Endpoint Tester")
    test_endpoint = st.selectbox(
        "Select Endpoint to Test:",
        [
            "GET / (System Status & Health)",
            "GET /api/potholes?city=Mangaluru (Fetch City Hazards)",
            "POST /api/trip/check-ahead (Warning Ahead Evaluation)",
            "GET /api/analytics/stats (Civic Safety Metrics)"
        ]
    )

    if st.button("🚀 Execute REST Request", type="primary"):
        try:
            if "GET / " in test_endpoint:
                target_endpoint = f"{api_url}/"
                req = urllib.request.Request(target_endpoint)
            elif "city=Mangaluru" in test_endpoint:
                target_endpoint = f"{api_url}/api/potholes?city=Mangaluru"
                req = urllib.request.Request(target_endpoint)
            elif "/api/analytics/stats" in test_endpoint:
                target_endpoint = f"{api_url}/api/analytics/stats"
                req = urllib.request.Request(target_endpoint)
            elif "check-ahead" in test_endpoint:
                target_endpoint = f"{api_url}/api/trip/check-ahead"
                payload = json.dumps({"lat": 12.8724, "lon": 74.8573, "heading": 245.0}).encode("utf-8")
                req = urllib.request.Request(target_endpoint, data=payload, headers={"Content-Type": "application/json"})

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=3.0) as r:
                res_body = json.loads(r.read().decode())
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            st.success(f"Response: HTTP 200 OK ({latency_ms} ms)")
            st.json(res_body)
        except Exception as ex:
            st.error(f"Request failed: {str(ex)}")

# ------------------------------------------------------------------------------
# TAB 6: PBL VIVA DEFENSE & ACADEMIC SPECS
# ------------------------------------------------------------------------------
with tab_pbl:
    st.markdown("### 🎓 Academic Specifications & PBL Viva Defense")
    st.write("Formal definitions, mathematical formulations, and viva defense reference for project evaluations.")

    col_pbl1, col_pbl2 = st.columns(2)

    with col_pbl1:
        st.markdown("""
        #### 📐 Mathematical Formulations
        1. **Haversine Distance Formulation:**
           $$d = 2 R \\arcsin \\left( \\sqrt{\\sin^2\\left(\\frac{\\Delta \\phi}{2}\\right) + \\cos(\\phi_1)\\cos(\\phi_2)\\sin^2\\left(\\frac{\\Delta \\lambda}{2}\\right)} \\right)$$
           Used for 15m spatial deduplication and 250m driver warning radius ($R = 6371000\\text{ m}$).

        2. **Compass Bearing Formulation:**
           $$\\theta = \\text{atan2}\\left(\\sin(\\Delta \\lambda)\\cos(\\phi_2), \\cos(\\phi_1)\\sin(\\phi_2) - \\sin(\\phi_1)\\cos(\\phi_2)\\cos(\\Delta \\lambda)\\right)$$
           Normalised to $[0^\\circ, 360^\\circ)$.

        3. **Forward Directional Cone Check:**
           $$\\delta = |\\theta_{\\text{heading}} - \\theta_{\\text{hazard}}| \\pmod{360} \\le 45^\\circ$$
           Eliminates false alarms for potholes located behind the driver or on parallel opposing carriageways.
        """)

    with col_pbl2:
        st.markdown("""
        #### 🔍 Architectural Comparison
        | Dimension | Ultrasonic/Accelerometer Sensor (Prior Work) | RideVision (CV + Spatial AI) |
        | :--- | :--- | :--- |
        | **Hardware** | Custom Arduino/PIC + external sensors | Commodity Smartphone (Android / iOS) |
        | **Visual Proof** | None (Blind numeric depth spike) | Bounding-box annotated photographic evidence |
        | **Repair Tracking**| None (Fixed potholes alert forever) | Crowdsourced confirmation with 7-day cooldown |
        | **Civic Forwarding**| Generic SMS | Direct WhatsApp (`wa.me`) & BBMP Helpline |
        """)

    st.markdown("---")
    st.markdown("#### 🎯 Viva Defense Prepared Q&A")
    with st.expander("Q1: Why choose YOLOv8 nano rather than heavier detectors like YOLOv8x or Faster R-CNN?"):
        st.write("YOLOv8 nano (~3.2M parameters, < 6.5 MB weights) achieves 30+ FPS on edge smartphones without triggering device thermal throttling or draining excessive battery during daily commutes.")

    with st.expander("Q2: How does the system handle rain, puddles, and nighttime lighting?"):
        st.write("Water reflections are known computer vision edge cases. RideVision applies multi-frame temporal confirmation before filing an automatic report, combined with crowdsourced verification.")

    with st.expander("Q3: How does the spatial deduplication engine prevent duplicate complaints?"):
        st.write("Whenever a report is filed, the database computes the Haversine distance to all unresolved potholes in that city. If an existing record lies within 15 meters, the system merges the report into that master hazard rather than spamming municipal authorities.")

    with st.expander("Q4: What stops malicious users from spamming fake 'Fixed' votes?"):
        st.write("The anti-gaming guard enforces a strict 7-day cooldown per user per hazard in SQLite, rejecting repeat votes until the window expires.")
