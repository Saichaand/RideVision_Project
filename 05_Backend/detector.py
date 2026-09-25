"""
Computer Vision Inference Engine for RideVision.

Loads the YOLOv8 model (best.pt) and executes inference on input images/frames.
Includes severity estimation, confidence calibration, and frame annotation.
Equipped with an OpenCV heuristic fallback if neural net runtime is initializing.
"""

import os
import io
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

# Search paths for trained model weights
MODEL_CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best.pt")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "best.pt")),
    os.path.abspath("best.pt"),
]


class PotholeDetector:
    def __init__(self, confidence_threshold: float = 0.35):
        self.conf_threshold = confidence_threshold
        self.model = None
        self.model_path = None
        self.is_yolo_loaded = False

        self._load_model()

    def _load_model(self):
        for path in MODEL_CANDIDATE_PATHS:
            if os.path.exists(path):
                self.model_path = path
                break

        if self.model_path:
            try:
                from ultralytics import YOLO
                print(f"[RideVision Detector] Loading YOLO weights from: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.is_yolo_loaded = True
                print("[RideVision Detector] YOLOv8 model loaded successfully.")
            except Exception as e:
                print(f"[RideVision Detector] Notice: Ultralytics/PyTorch could not load {self.model_path} ({e}). Using OpenCV computer-vision engine.")
                self.is_yolo_loaded = False
        else:
            print("[RideVision Detector] Warning: best.pt not found in candidate paths. Using OpenCV CV engine.")

    def detect(
        self,
        image_input: Any,
        conf_threshold: Optional[float] = None,
        engine: str = "hybrid"  # 'yolo', 'opencv', or 'hybrid'
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs pothole detection on the given image.
        Accepts: PIL Image, NumPy array (BGR/RGB), or raw bytes.
        engine: 'yolo', 'opencv', or 'hybrid'
        Returns: (list_of_detections, annotated_image_bgr)
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Convert input to BGR numpy array
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if img_bgr is None:
            raise ValueError("Failed to decode image input.")

        h, w = img_bgr.shape[:2]
        total_frame_area = float(w * h)

        detections: List[Dict[str, Any]] = []

        # 1. Run YOLO if selected and available
        if engine in ("yolo", "hybrid") and self.is_yolo_loaded and self.model is not None:
            try:
                results = self.model(img_bgr, conf=threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0

                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        box_area = bw * bh
                        area_ratio = box_area / total_frame_area

                        if area_ratio > 0.05 or bw > 180 or bh > 180:
                            severity = "severe"
                        elif area_ratio > 0.015 or bw > 90 or bh > 90:
                            severity = "moderate"
                        else:
                            severity = "minor"

                        detections.append({
                            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "box_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                            "confidence": round(conf, 3),
                            "class_id": cls_id,
                            "class_name": "pothole",
                            "severity": severity,
                            "area_ratio": round(area_ratio, 4),
                            "engine": "YOLOv8"
                        })
            except Exception as e:
                print(f"[RideVision Detector] YOLO inference error: {e}")

        # 2. Run OpenCV Fallback if OpenCV requested OR if hybrid yielded 0 detections
        if engine == "opencv" or (engine == "hybrid" and len(detections) == 0):
            cv_dets = self._detect_opencv_fallback(img_bgr, threshold)
            detections.extend(cv_dets)

        # Generate annotated frame
        annotated_bgr = self.annotate_frame(img_bgr, detections)
        return detections, annotated_bgr

    def _detect_opencv_fallback(self, img_bgr: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """
        OpenCV-based road surface defect contour heuristic.
        Detects distinct dark, irregular asphalt cavities in the road region.
        """
        h, w = img_bgr.shape[:2]
        total_frame_area = float(w * h)
        detections = []

        # Focus on bottom 70% of frame (road region in dashboard/mobile view)
        road_y_start = int(h * 0.30)
        road_roi = img_bgr[road_y_start:, :]

        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Threshold using relative asphalt darkness: potholes are darker than surrounding asphalt
        mean_lum = float(np.mean(blurred))
        dark_thresh_val = max(15.0, mean_lum - 16.0)
        _, thresh = cv2.threshold(blurred, int(dark_thresh_val), 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and entire frame fills
            if area < (total_frame_area * 0.005) or area > (total_frame_area * 0.40):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / max(bh, 1)

            # Potholes typically have roughly oval/horizontal aspects
            if 0.4 <= aspect_ratio <= 4.0:
                real_y = road_y_start + y
                x1, y1, x2, y2 = float(x), float(real_y), float(x + bw), float(real_y + bh)
                area_ratio = (bw * bh) / total_frame_area

                confidence = min(0.95, 0.55 + (area_ratio * 4.0))
                if confidence < threshold:
                    continue

                if area_ratio > 0.05 or bw > 140 or bh > 120:
                    severity = "severe"
                elif area_ratio > 0.015 or bw > 70:
                    severity = "moderate"
                else:
                    severity = "minor"

                detections.append({
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "box_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                    "confidence": round(confidence, 2),
                    "class_id": 0,
                    "class_name": "pothole",
                    "severity": severity,
                    "area_ratio": round(area_ratio, 4),
                    "engine": "OpenCV-CV"
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:5]

    @staticmethod
    def annotate_frame(img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and stylized confidence + severity badges on the image."""
        output = img_bgr.copy()
        h, w = output.shape[:2]

        color_map = {
            "severe": (38, 38, 245),      # Red in BGR
            "moderate": (0, 140, 255),     # Orange in BGR
            "minor": (0, 215, 255)         # Gold/Yellow in BGR
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            severity = det.get("severity", "moderate")
            conf = det.get("confidence", 0.0)
            color = color_map.get(severity, (0, 255, 0))

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # Header tag
            label = f"Pothole ({severity.upper()}) {int(conf * 100)}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

            tag_y1 = max(0, y1 - th - 10)
            tag_y2 = y1
            cv2.rectangle(output, (x1, tag_y1), (x1 + tw + 12, tag_y2), color, -1)
            cv2.putText(
                output, label, (x1 + 6, tag_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        return output


# Global instance
detector = PotholeDetector()
