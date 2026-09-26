"""
RideVision — Dual-Mode Road Safety & Municipal Engineering Portal.
St Joseph Engineering College (SJEC) – Dept. of AIML, PBL Project.

Modes:
  1. 👤 Commuter Mode (User): Clean, intuitive, driver-centric interface for live HUD alerts,
     1-click hazard reporting with WhatsApp dispatch, safe route map, and crowd verification.
  2. 🛠️ Dev & Admin Console: Technical workbench for CV inference benchmarks, SQLite database
     management, REST API telemetry, and academic PBL viva specifications.
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
import streamlit.components.v1 as components
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
    route_complaint
)
from detector import detector

# Ensure database is initialized
init_db()

# Page configuration
st.set_page_config(
    page_title="RideVision | Road Safety & Pothole System",
    page_icon="🚗",
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
        background: linear-gradient(135deg, #FF6B4A 0%, #FFA726 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-title.dev-mode {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    .badge-user {
        background: rgba(255, 107, 74, 0.15);
        color: #FF8A65;
        border: 1px solid rgba(255, 107, 74, 0.35);
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

    /* WhatsApp & Action Buttons */
    .wa-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
        color: white !important;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 10px;
        text-decoration: none !important;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(37, 211, 102, 0.35);
        transition: transform 0.15s ease;
    }
    .wa-button:hover {
        transform: translateY(-2px);
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
# SIDEBAR: ROLE / MODE SELECTION & HIGH-LEVEL TELEMETRY
# ==============================================================================
st.sidebar.markdown("### 🚗 **RideVision**")
st.sidebar.caption("CV Pothole Detection & Civic Safety System")
st.sidebar.markdown("---")

# User vs Dev Mode Switcher
app_mode = st.sidebar.radio(
    "🎯 Select Workspace Mode:",
    ["👤 Commuter Mode (User)", "🛠️ Dev & Admin Console"],
    index=0,
    help="Switch between clean commuter-friendly driving & reporting view, or technical engineering & evaluation console."
)

is_dev = "Dev & Admin" in app_mode

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
detector_status = "🟢 YOLOv8 Loaded" if detector.is_yolo_loaded else "🟡 OpenCV Fallback"
st.sidebar.caption(f"**Vision Engine:** {detector_status}")
st.sidebar.caption("St Joseph Engineering College (SJEC) – AIML")

# ==============================================================================
# APP HEADER
# ==============================================================================
if not is_dev:
    # Commuter Mode Header
    st.markdown("""
    <div class="hero-container">
        <div>
            <div class="hero-title">RideVision 🚗💨 <span class="mode-badge badge-user">Commuter Mode</span></div>
            <div class="hero-subtitle">Proactive 250m Road Hazard Alerts • 1-Click MCC WhatsApp Grievance Dispatch • Crowd-Verified Roads</div>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.85rem; color: #10B981; font-weight: 600;">● SYSTEM ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Dev Mode Header
    st.markdown("""
    <div class="hero-container">
        <div>
            <div class="hero-title dev-mode">RideVision Engineering Console 🛠️ <span class="mode-badge badge-dev">Developer & Admin</span></div>
            <div class="hero-subtitle">YOLOv8 Edge Inference Benchmarks • SQLite Spatial DB • REST API Telemetry • PBL Viva Defense</div>
        </div>
        <div style="text-align: right;">
            <span class="code-tag" style="color: #38BDF8;">FastAPI :8000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: COMMUTER (USER) MODE
# Simple, intuitive, focused on driving safety and quick reporting
# ==============================================================================
if not is_dev:
    u_tab_hud, u_tab_report, u_tab_map, u_tab_verify = st.tabs([
        "🛡️ Drive & Hazard HUD",
        "📸 Snap & Report Pothole",
        "🗺️ Road Safety Map",
        "🗳️ Community Verification"
    ])

    # --------------------------------------------------------------------------
    # USER TAB 1: DRIVE & HAZARD HUD
    # --------------------------------------------------------------------------
    with u_tab_hud:
        st.markdown("### 🛡️ Live Driver Head-Up Display (HUD)")
        st.write("Real-time safety navigation: As you drive along your commute, RideVision continuously scans for confirmed potholes within **250 meters ahead** in your travel lane and triggers early directional alerts.")

        col_sim_ctrl, col_hud_display = st.columns([1.1, 1.4])

        with col_sim_ctrl:
            st.markdown("#### 🛣️ Commute Route & Navigation")
            route_mode = st.radio("Route Selection:", ["Popular City Commutes", "Custom Origin & Destination"], horizontal=True)

            preset_routes = {
                "Mangaluru — SJEC Vamanjoor to Kankanady (NH 73)": {
                    "start": (12.9152, 74.8988, "SJEC Vamanjoor Gate"),
                    "end": (12.8715, 74.8564, "Kankanady Junction"),
                    "city": "Mangaluru"
                },
                "Mangaluru — Kadri Mallikatte to Hampankatta": {
                    "start": (12.8798, 74.8532, "Kadri Mallikatte"),
                    "end": (12.8646, 74.8425, "Hampankatta City Center"),
                    "city": "Mangaluru"
                },
                "Bengaluru — Indiranagar 100ft Road to Domlur": {
                    "start": (12.9719, 77.6412, "Indiranagar 100ft Road"),
                    "end": (12.9610, 77.6410, "Domlur Flyover"),
                    "city": "Bengaluru"
                },
                "Udupi — Manipal Tiger Circle to City Bus Stand": {
                    "start": (13.3525, 74.7865, "Manipal Tiger Circle"),
                    "end": (13.3408, 74.7421, "Udupi Bus Stand"),
                    "city": "Udupi"
                }
            }

            if route_mode == "Popular City Commutes":
                sel_route_label = st.selectbox("Select Commute Corridor:", list(preset_routes.keys()))
                route_meta = preset_routes[sel_route_label]
                start_lat, start_lon, start_name = route_meta["start"]
                end_lat, end_lon, end_name = route_meta["end"]
            else:
                c_start_in, c_end_in = st.columns(2)
                with c_start_in:
                    custom_start = st.text_input("Start Location:", value="SJEC Vamanjoor, Mangaluru")
                with c_end_in:
                    custom_dest = st.text_input("Destination:", value="Kankanady, Mangaluru")

                if st.button("📍 Geocode & Trace Route", use_container_width=True):
                    g_start = geocode_address(custom_start)
                    g_dest = geocode_address(custom_dest)
                    if g_start and g_dest:
                        st.session_state["custom_route_coords"] = (
                            g_start["lat"], g_start["lon"], g_start["road"],
                            g_dest["lat"], g_dest["lon"], g_dest["road"]
                        )
                        st.success(f"Route traced: {g_start['road']} ➔ {g_dest['road']}")
                    else:
                        st.warning("Could not geocode one of the addresses. Using default coordinates.")

                if "custom_route_coords" in st.session_state:
                    start_lat, start_lon, start_name, end_lat, end_lon, end_name = st.session_state["custom_route_coords"]
                else:
                    start_lat, start_lon, start_name = (12.9152, 74.8988, "SJEC Vamanjoor Gate")
                    end_lat, end_lon, end_name = (12.8715, 74.8564, "Kankanady Junction")

            # Route Calculation (cached in session state per endpoints)
            route_key = f"osrm_{start_lat:.4f}_{start_lon:.4f}_{end_lat:.4f}_{end_lon:.4f}"
            if route_key not in st.session_state:
                with st.spinner("Computing real driving route via OSRM..."):
                    st.session_state[route_key] = get_route_osrm(start_lat, start_lon, end_lat, end_lon)

            route_data = st.session_state[route_key]
            route_coords = route_data["coordinates"]
            total_pts = len(route_coords)

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                    <span>🚩 <b>Start:</b> {start_name}</span>
                    <span>🏁 <b>Dest:</b> {end_name}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: #38BDF8; margin-top: 6px;">
                    <span>🛣️ Distance: <b>{route_data['distance_km']} km</b></span>
                    <span>⏱️ Est. Drive Time: <b>{route_data['duration_min']} mins</b></span>
                    <span>📍 Waypoints: <b>{total_pts}</b></span>
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
                st.warning(f"⚠️ **Hazard Notice:** Found {len(route_potholes)} active pothole(s) along this commute road.")
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
                        👥 <b>Community Confirmations:</b> {warning['confirmation_count']} commuter vote(s)
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
                {"lat": curr_lat, "lon": curr_lon, "name": "Your Vehicle", "color": [0, 210, 255, 255], "radius": 40}
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

    # --------------------------------------------------------------------------
    # USER TAB 2: SNAP & REPORT POTHOLE
    # --------------------------------------------------------------------------
    with u_tab_report:
        st.markdown("### 📸 Quick Road Hazard Report")
        st.write("Snap or select a road photo. Our on-device Computer Vision rates damage severity, detects your exact geographical location & road jurisdiction, and generates an official complaint for municipal grievance desks.")

        # HTML5 Device Geolocation receiver
        if "user_gps" in st.query_params:
            try:
                g_lat, g_lon = map(float, st.query_params["user_gps"].split(","))
                st.session_state["report_lat"] = g_lat
                st.session_state["report_lon"] = g_lon
                rev = reverse_geocode(g_lat, g_lon)
                st.session_state["report_landmark"] = rev["road"]
                st.session_state["report_city"] = rev["city"]
                del st.query_params["user_gps"]
                st.rerun()
            except Exception:
                pass

        if "report_lat" not in st.session_state:
            st.session_state["report_lat"] = 12.9152
        if "report_lon" not in st.session_state:
            st.session_state["report_lon"] = 74.8988
        if "report_landmark" not in st.session_state:
            st.session_state["report_landmark"] = "NH 73 near SJEC, Vamanjoor"
        if "report_city" not in st.session_state:
            st.session_state["report_city"] = "Mangaluru"

        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        
        rep_col1, rep_col2 = st.columns([1.1, 1.3])

        with rep_col1:
            st.markdown("#### 1. Select or Upload Road Photo")
            img_mode = st.radio("Image Source:", ["Use Benchmark / Dataset Sample", "Upload My Photo"], horizontal=True)

            user_image = None
            if img_mode == "Use Benchmark / Dataset Sample":
                # Combine standard samples with real dataset images
                # Only use real dataset images if available
                sample_picks = {
                    "🔴 Severe Pothole (NH 73 Mangaluru)": os.path.join(samples_dir, "sample_severe_pothole.jpg"),
                    "🟠 Moderate Cavity (City Asphalt)": os.path.join(samples_dir, "sample_moderate_pothole.jpg"),
                    "🟢 Clean Resurfaced Road": os.path.join(samples_dir, "sample_clean_road.jpg")
                }
                # Add real dataset images if available
                dataset_images = get_dataset_samples(limit=25)
                sample_picks.update(dataset_images)

                pick_label = st.selectbox("Pick a Road Frame:", list(sample_picks.keys()))
                path = sample_picks[pick_label]
                if os.path.exists(path):
                    user_image = Image.open(path)
            else:
                up_file = st.file_uploader("Upload road photo (JPG/PNG):", type=["jpg", "jpeg", "png"])
                if up_file:
                    user_image = Image.open(up_file)

            if user_image:
                st.image(user_image, caption="Road Frame Input", use_container_width=True)

        with rep_col2:
            st.markdown("#### 2. AI Severity & Computer Vision")
            if user_image:
                with st.spinner("Analyzing road frame with YOLOv8..."):
                    detections, annotated_bgr = detector.detect(user_image, conf_threshold=0.30, engine="hybrid")
                    annotated_rgb = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)) if 'cv2' in sys.modules else user_image

                st.image(annotated_rgb, caption=f"AI Vision Assessment: Found {len(detections)} hazard cavity(ies)", use_container_width=True)

                if detections:
                    severities = [d["severity"] for d in detections]
                    top_sev = "severe" if "severe" in severities else ("moderate" if "moderate" in severities else "minor")
                    sev_colors = {"severe": "🔴 SEVERE CAVITY", "moderate": "🟠 MODERATE CAVITY", "minor": "🟡 MINOR ABRASION"}
                    st.success(f"**AI Assessment Verdict:** {sev_colors[top_sev]} (Max confidence: {int(detections[0]['confidence']*100)}%)")
                else:
                    top_sev = "minor"
                    st.info("Road frame has subtle abrasions or smooth surface. You can still report it.")

                # Location & Civic Dispatch Form
                st.markdown("#### 3. Real Location & Municipal Authority")

                # Location Detection Tools
                loc_tool_col1, loc_tool_col2 = st.columns(2)
                with loc_tool_col1:
                    if st.button("🌐 Auto-Detect via Network IP", use_container_width=True):
                        ip_loc = detect_ip_location()
                        st.session_state["report_lat"] = ip_loc["lat"]
                        st.session_state["report_lon"] = ip_loc["lon"]
                        rev = reverse_geocode(ip_loc["lat"], ip_loc["lon"])
                        st.session_state["report_landmark"] = rev["road"]
                        st.session_state["report_city"] = rev["city"]
                        st.success(f"📍 Location detected: {rev['road']}, {rev['city']}")
                        st.rerun()

                with loc_tool_col2:
                    # HTML5 Browser Geolocation trigger
                    components.html("""
                    <button onclick="getGPS()" style="
                        width: 100%; height: 38px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15);
                        background: linear-gradient(135deg, #1E293B, #0F172A); color: #38BDF8; font-weight: 600;
                        font-size: 0.88rem; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        🎯 Get Browser GPS
                    </button>
                    <script>
                    function getGPS() {
                        if (navigator.geolocation) {
                            navigator.geolocation.getCurrentPosition(function(pos) {
                                const lat = pos.coords.latitude.toFixed(5);
                                const lon = pos.coords.longitude.toFixed(5);
                                const pUrl = new URL(window.parent.location.href);
                                pUrl.searchParams.set('user_gps', lat + ',' + lon);
                                window.parent.location.href = pUrl.href;
                            }, function(err) {
                                alert('GPS access error: ' + err.message);
                            }, {enableHighAccuracy: true, timeout: 6000});
                        } else {
                            alert('Geolocation not supported by this browser.');
                        }
                    }
                    </script>
                    """, height=44)

                # Landmark / Address Search
                search_col1, search_col2 = st.columns([3, 1])
                with search_col1:
                    srch_query = st.text_input("🔍 Search Landmark / Road Name:", placeholder="e.g. Kankanady Mangalore, SJEC Vamanjoor", label_visibility="collapsed")
                with search_col2:
                    if st.button("Search", use_container_width=True):
                        if srch_query:
                            geo_res = geocode_address(srch_query)
                            if geo_res:
                                st.session_state["report_lat"] = geo_res["lat"]
                                st.session_state["report_lon"] = geo_res["lon"]
                                st.session_state["report_landmark"] = geo_res["road"]
                                st.session_state["report_city"] = geo_res["city"]
                                st.success(f"Found: {geo_res['display_name'][:60]}")
                                st.rerun()
                            else:
                                st.warning("Landmark not found.")

                with st.form("quick_report_form"):
                    coord_c1, coord_c2 = st.columns(2)
                    with coord_c1:
                        f_lat = st.number_input("Latitude:", value=float(st.session_state["report_lat"]), format="%.5f")
                    with coord_c2:
                        f_lon = st.number_input("Longitude:", value=float(st.session_state["report_lon"]), format="%.5f")

                    # Auto reverse-geocode city and road
                    det_city = get_city_from_coords(f_lat, f_lon)
                    city_opts = ["Mangaluru", "Bengaluru", "Udupi", "Mysuru"]
                    if det_city not in city_opts:
                        city_opts.insert(0, det_city)
                    city_idx = city_opts.index(det_city) if det_city in city_opts else 0

                    f_city = st.selectbox("Municipal Jurisdiction (Auto-Identified):", city_opts, index=city_idx)
                    f_landmark = st.text_input("Road Landmark / Address:", value=st.session_state["report_landmark"])
                    f_note = st.text_input("Commuter Observation / Note:", value="Dangerous cavity depth for two-wheelers and night commuters")

                    # Live authority preview
                    city_cfg = get_city_config(f_city)
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3); margin-top: 8px;">
                        <span style="font-size: 0.85rem; color: #94A3B8;">🏛️ Official Redressal Authority:</span>
                        <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">{city_cfg.get('authority_name')}</div>
                        <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">Channel: <b>{city_cfg.get('channel_type', 'whatsapp').upper()}</b> • Contact: <b>{city_cfg.get('contact_value')}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                    send_btn = st.form_submit_button("🚀 Submit Verified Hazard Report", type="primary", use_container_width=True)

                    if send_btn:
                        res = save_or_merge_pothole(
                            lat=f_lat, lon=f_lon, city=f_city, severity=top_sev,
                            user_id="commuter_user_app", address=f_landmark, note=f_note
                        )
                        pothole_obj = res["pothole"]
                        route_res = route_complaint(pothole_obj, city_cfg)

                        # Set focus for Tab 3 map
                        st.session_state["focus_lat"] = f_lat
                        st.session_state["focus_lon"] = f_lon
                        st.session_state["focus_pothole_id"] = pothole_obj["id"]

                        st.balloons()
                        st.success(f"✅ Hazard registered in SQLite database! Hazard ID: `{pothole_obj['id']}` at coordinates ({f_lat:.5f}, {f_lon:.5f})")
                        if not res["is_new"]:
                            st.info("ℹ️ Automatically merged with existing report within 15m radius to prevent duplicate civic tickets.")

                        # Show Dispatch Action
                        if route_res.get("channel") == "whatsapp" and route_res.get("link"):
                            st.markdown(f"""
                            <div style="margin-top: 14px; text-align: center; background: rgba(37, 211, 102, 0.15); padding: 14px; border-radius: 10px; border: 1px solid #25D366;">
                                <a class="wa-button" href="{route_res['link']}" target="_blank" style="
                                    display: inline-block; padding: 10px 22px; background: #25D366; color: #FFFFFF; font-weight: 700;
                                    border-radius: 8px; text-decoration: none; font-size: 1rem; box-shadow: 0 4px 12px rgba(37, 211, 102, 0.35);">
                                    💬 Forward to {route_res['authority']} WhatsApp Grievance Desk
                                </a>
                                <p style="font-size:0.84rem; color: #E2E8F0; margin-top: 8px; margin-bottom: 0;">
                                    Pre-fills official complaint with exact GPS Google Maps link, photos, and severity.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif route_res.get("channel") == "helpline":
                            st.markdown(f"""
                            <div style="margin-top: 14px; padding: 12px; background: rgba(59, 130, 246, 0.2); border-radius: 8px; border: 1px solid #3B82F6;">
                                📞 <b>Official Dispatch:</b> Forward to {route_res['authority']} Helpline: <b>{route_res['contact']}</b>
                            </div>
                            """, unsafe_allow_html=True)

                        # Mini map of exact pothole placement
                        st.markdown("##### 📍 Verified Map Placement:")
                        mini_df = pd.DataFrame([{"lat": f_lat, "lon": f_lon, "color": [239, 68, 68, 240]}])
                        st.pydeck_chart(pdk.Deck(
                            map_style="road",
                            initial_view_state=pdk.ViewState(latitude=f_lat, longitude=f_lon, zoom=15, pitch=25),
                            layers=[pdk.Layer("ScatterplotLayer", mini_df, get_position="[lon, lat]", get_color="color", get_radius=30)]
                        ))
            else:
                st.caption("Please select a sample road image or upload a photo to start.")

    # --------------------------------------------------------------------------
    # USER TAB 3: ROAD SAFETY MAP
    # --------------------------------------------------------------------------
    with u_tab_map:
        st.markdown("### 🗺️ Live Road Safety Map")
        st.write("Browse verified hazards and repaired stretches in your city before starting your commute.")

        # Focus banner if newly reported
        if "focus_pothole_id" in st.session_state:
            st.info(f"📍 **Focusing on your newly reported pothole** (ID: `{st.session_state['focus_pothole_id']}`) at coordinates: {st.session_state.get('focus_lat', 0):.5f}, {st.session_state.get('focus_lon', 0):.5f}")

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
            focus_id = st.session_state.get("focus_pothole_id")
            for p in all_p_map:
                is_focused = (focus_id and p["id"] == focus_id)
                if is_focused:
                    color = [255, 0, 128, 255]   # Neon Pink for recently submitted
                    label_status = "JUST REPORTED"
                    radius = 90
                elif p["status"] == "verified_fixed":
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

            # Center map on focused pothole if available, else center on mean
            if "focus_lat" in st.session_state and "focus_lon" in st.session_state and c_arg is None:
                view_lat = float(st.session_state["focus_lat"])
                view_lon = float(st.session_state["focus_lon"])
                view_zoom = 14
            else:
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
                    "html": "<b>{address}</b><br>City: {city}<br>Condition: <b>{status}</b><br>Votes: {confirmations}<br><a href='{maps_link}' target='_blank' style='color:#38BDF8;'>Open in Google Maps ↗</a>",
                    "style": {"backgroundColor": "#0F172A", "color": "#FFFFFF", "borderRadius": "8px", "padding": "8px"}
                }
            )
            st.pydeck_chart(view_deck)

            # Quick Card Summary
            st.markdown("#### 📍 Hazard Stretches")
            st.dataframe(
                df_p_map[["city", "address", "severity", "status", "confirmations"]],
                use_container_width=True
            )
        else:
            st.info("No road hazards found matching the selected filter.")

    # --------------------------------------------------------------------------
    # USER TAB 4: COMMUNITY VERIFICATION
    # --------------------------------------------------------------------------
    with u_tab_verify:
        st.markdown("### 🗳️ Commuter Verification")
        st.write("Help your fellow commuters keep road data accurate. If you recently passed a known pothole, vote whether it is still there or has been repaired.")

        all_pot_verify = get_all_potholes()
        if all_pot_verify:
            v_col1, v_col2 = st.columns([1.2, 1])

            with v_col1:
                v_options = {f"{p['address']} ({p['city']}) — Status: {p['status'].upper()}": p for p in all_pot_verify}
                chosen_label = st.selectbox("Select Hazard You Passed By:", list(v_options.keys()))
                target_pot = v_options[chosen_label]

                commuter_id = st.text_input("Your Commuter ID:", value="daily_rider_sjec")
                st.caption("🛡️ **Anti-Spam Safeguard:** Each commuter can cast 1 confirmation per hazard every 7 days.")

            with v_col2:
                st.markdown("#### Hazard Card")
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">{target_pot.get('address')}</div>
                    <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px;">City: {target_pot['city']} • Severity: {target_pot['severity'].upper()}</div>
                    <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>⚠️ 'Still There' Votes: <b>{target_pot['confirmation_count']}</b></span>
                        <span>✅ 'Fixed' Votes: <b>{target_pot['fixed_confirmation_count']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                last_confirmed = get_last_user_confirmation(commuter_id, target_pot["id"])
                if last_confirmed:
                    st.caption(f"Last confirmed by you: {last_confirmed.strftime('%Y-%m-%d %H:%M')} UTC")

                btn_v1, btn_v2 = st.columns(2)
                with btn_v1:
                    v_still = st.button("⚠️ It's Still There", use_container_width=True)
                with btn_v2:
                    v_fixed = st.button("✅ It's Repaired!", use_container_width=True, type="primary")

                if v_still or v_fixed:
                    v_type = "still_there" if v_still else "fixed"
                    if is_confirmation_eligible(commuter_id, target_pot["id"], last_confirmed, cooldown_days=7):
                        record_confirmation(commuter_id, target_pot["id"], v_type)
                        st.success(f"Thank you! Your vote '{v_type}' has been recorded.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("⛔ **7-Day Cooldown Active:** You have already confirmed this spot recently. This safeguard ensures crowdsourced data cannot be spammed.")


# ==============================================================================
# SECTION 2: DEV & ADMIN CONSOLE (ENGINEER MODE)
# In-depth technical workbench for evaluation, testing, and system inspection
# ==============================================================================
else:
    dev_tab_model, dev_tab_db, dev_tab_api, dev_tab_pbl = st.tabs([
        "🧠 CV Model & Inference Benchmarks",
        "🏛️ Database & Civic Routing Admin",
        "⚡ REST API Telemetry & Tester",
        "🎓 PBL Viva Defense & Academic Specs"
    ])

    # --------------------------------------------------------------------------
    # DEV TAB 1: CV MODEL & INFERENCE LAB
    # --------------------------------------------------------------------------
    with dev_tab_model:
        st.markdown("### 🧠 Computer Vision & Edge Model Benchmarks")
        st.write("Inspect neural network inference, test YOLOv8 vs OpenCV contour fallback, adjust hyper-parameters, and inspect bounding box tensor metrics.")

        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        col_dev_img, col_dev_res = st.columns([1, 1.4])

        with col_dev_img:
            st.markdown("#### 1. Input Test Frame")
            bench_picks = {
                "Sample 1: Severe Road Cavity (sample_severe_pothole.jpg)": os.path.join(samples_dir, "sample_severe_pothole.jpg"),
                "Sample 2: Moderate Asphalt Cavity (sample_moderate_pothole.jpg)": os.path.join(samples_dir, "sample_moderate_pothole.jpg"),
                "Sample 3: Clean Road Control (sample_clean_road.jpg)": os.path.join(samples_dir, "sample_clean_road.jpg"),
            }
            # Include real dataset images if available
            bench_picks.update(get_dataset_samples(limit=25))

            test_img_pick = st.selectbox("Select Benchmark Frame:", list(bench_picks.keys()))
            loaded_img = Image.open(bench_picks[test_img_pick])
            st.image(loaded_img, caption="Benchmark Input Frame (640x640 normalized)", use_container_width=True)

            st.markdown("#### 2. Inference Hyperparameters")
            dev_conf = st.slider("Confidence Cutoff Threshold:", 0.05, 0.95, 0.30, 0.05)
            dev_engine = st.selectbox("Inference Engine Runtime:", ["hybrid", "yolo", "opencv"])

        with col_dev_res:
            st.markdown("#### 3. Execution Telemetry")
            t_start = time.perf_counter()
            detections, ann_bgr = detector.detect(loaded_img, conf_threshold=dev_conf, engine=dev_engine)
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

    # --------------------------------------------------------------------------
    # DEV TAB 2: DATABASE & CIVIC ROUTING ADMIN
    # --------------------------------------------------------------------------
    with dev_tab_db:
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

    # --------------------------------------------------------------------------
    # DEV TAB 3: REST API TELEMETRY & TESTER
    # --------------------------------------------------------------------------
    with dev_tab_api:
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

    # --------------------------------------------------------------------------
    # DEV TAB 4: PBL VIVA DEFENSE & ACADEMIC SPECS
    # --------------------------------------------------------------------------
    with dev_tab_pbl:
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
