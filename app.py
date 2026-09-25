"""
RideVision Root Launcher for Streamlit Dashboard.
Directs to 06_Web_Dashboard/app.py.
"""
import os
import runpy


dashboard_path = os.path.join(os.path.dirname(__file__), "06_Web_Dashboard", "app.py")
runpy.run_path(dashboard_path, run_name="__main__")
