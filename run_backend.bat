@echo off
title RideVision — FastAPI Backend Server
echo ===================================================
echo   Starting RideVision FastAPI Backend Server...
echo ===================================================
python -m uvicorn main:app --app-dir 05_Backend --host 0.0.0.0 --port 8000 --reload
pause
