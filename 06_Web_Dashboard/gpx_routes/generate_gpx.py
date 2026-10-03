"""
Generates real-world GPX (GPS Exchange Format) route files for RideVision commute tracks.
"""

import os
import sys
import xml.etree.ElementTree as ET

# Add 05_Backend to Python search path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "05_Backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from geo_matching import get_route_osrm

GPX_DIR = os.path.dirname(__file__)

ROUTES = [
    {
        "filename": "mangaluru_sjec_to_kankanady.gpx",
        "name": "Mangaluru — SJEC Vamanjoor to Kankanady (NH 73)",
        "desc": "Key Mangaluru arterial corridor along National Highway 73 via Vamanjoor, Bikarnakatte, and Kankanady.",
        "start": (12.9152, 74.8988),
        "end": (12.8715, 74.8564)
    },
    {
        "filename": "mangaluru_kadri_to_hampankatta.gpx",
        "name": "Mangaluru — Kadri Mallikatte to Hampankatta",
        "desc": "Urban city corridor passing Kadri Temple Road, Bunts Hostel Circle, and Hampankatta Market.",
        "start": (12.8798, 74.8532),
        "end": (12.8646, 74.8425)
    },
    {
        "filename": "bengaluru_indiranagar_to_domlur.gpx",
        "name": "Bengaluru — Indiranagar 100ft Road to Domlur Flyover",
        "desc": "High-density tech corridor traversing 100 Feet Road, 12th Main junction, and Domlur EGL Flyover.",
        "start": (12.9719, 77.6412),
        "end": (12.9610, 77.6410)
    },
    {
        "filename": "udupi_manipal_to_busstand.gpx",
        "name": "Udupi — Manipal Tiger Circle to City Bus Stand",
        "desc": "Coastal district highway connecting Manipal University Campus to Udupi Central Bus Station.",
        "start": (13.3525, 74.7865),
        "end": (13.3408, 74.7421)
    }
]


def create_gpx_files():
    for r in ROUTES:
        res = get_route_osrm(r["start"][0], r["start"][1], r["end"][0], r["end"][1])
        coords = res.get("coordinates", [])

        gpx_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="RideVision GPS Engine" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata>
    <name>{r["name"]}</name>
    <desc>{r["desc"]}</desc>
  </metadata>
  <trk>
    <name>{r["name"]}</name>
    <trkseg>
'''
        for pt in coords:
            gpx_xml += f'      <trkpt lat="{pt[0]:.6f}" lon="{pt[1]:.6f}">\n        <ele>24.5</ele>\n      </trkpt>\n'

        gpx_xml += '''    </trkseg>
  </trk>
</gpx>'''

        out_path = os.path.join(GPX_DIR, r["filename"])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(gpx_xml)
        print(f"Generated GPX route: {r['filename']} ({len(coords)} waypoints, {res.get('distance_km')} km)")


if __name__ == "__main__":
    create_gpx_files()
