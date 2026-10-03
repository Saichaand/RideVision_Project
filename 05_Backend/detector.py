"""
Computer Vision Inference Engine for RideVision.

Dedicated YOLOv8 Deep Learning Pothole Detector (best.pt).
Executes edge neural network inference on input frames with:
  - Tightened Non-Maximum Suppression (NMS) IoU deduplication
  - Automatic containment & duplicate box suppression
  - Geometric surface-area damage severity classification (Severe / Moderate / Minor)
  - Visual bounding box and confidence tag annotation
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

# Search paths for trained YOLOv8 model weights
MODEL_CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best.pt")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "best.pt")),
    os.path.abspath("best.pt"),
]


class PotholeDetector:
    def __init__(self, confidence_threshold: float = 0.35, iou_threshold: float = 0.35):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
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
                print(f"[RideVision Detector] Loading YOLOv8 weights from: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.is_yolo_loaded = True
                print("[RideVision Detector] YOLOv8 model loaded successfully.")
            except Exception as e:
                print(f"[RideVision Detector] Error loading YOLOv8 model from {self.model_path}: {e}")
                self.is_yolo_loaded = False
        else:
            print("[RideVision Detector] Warning: best.pt not found in candidate paths.")

    @staticmethod
    def _suppress_overlapping_boxes(
        boxes: List[Tuple[float, float, float, float, float, int]],
        iou_thresh: float = 0.35,
        containment_thresh: float = 0.65
    ) -> List[Tuple[float, float, float, float, float, int]]:
        """
        Suppresses redundant, overlapping, or nested bounding boxes for the same pothole.
        Keeps higher-confidence boxes.
        """
        if not boxes:
            return []

        # Sort by confidence descending
        sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []

        for b in sorted_boxes:
            suppress = False
            b_w = max(0.0, b[2] - b[0])
            b_h = max(0.0, b[3] - b[1])
            b_area = b_w * b_h

            # Filter trivial noise slivers
            if b_w < 10 or b_h < 8:
                continue

            for k in keep:
                # Intersection
                ix1 = max(b[0], k[0])
                iy1 = max(b[1], k[1])
                ix2 = min(b[2], k[2])
                iy2 = min(b[3], k[3])
                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h

                if inter_area > 0:
                    k_w = max(0.0, k[2] - k[0])
                    k_h = max(0.0, k[3] - k[1])
                    k_area = k_w * k_h
                    union_area = b_area + k_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    min_area = min(b_area, k_area)
                    containment = inter_area / min_area if min_area > 0 else 0.0

                    if iou > iou_thresh or containment > containment_thresh:
                        suppress = True
                        break

            if not suppress:
                keep.append(b)

        return keep

    def detect(
        self,
        image_input: Any,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        engine: Optional[str] = None  # retained for backward compatibility
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs YOLOv8 pothole detection on the given image.
        Accepts: PIL Image, NumPy array (BGR/RGB), or raw image bytes.
        Returns: (list_of_detections, annotated_image_bgr)
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold

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

        # Execute YOLOv8 Neural Network Inference
        if self.is_yolo_loaded and self.model is not None:
            try:
                results = self.model(
                    img_bgr,
                    conf=threshold,
                    iou=iou_thresh,
                    agnostic_nms=True,
                    imgsz=640,
                    max_det=15,
                    verbose=False
                )
                raw_boxes = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0

                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        raw_boxes.append((x1, y1, x2, y2, conf, cls_id))

                # Post-NMS containment and deduplication suppression
                cleaned_boxes = self._suppress_overlapping_boxes(raw_boxes, iou_thresh=iou_thresh)

                for x1, y1, x2, y2, conf, cls_id in cleaned_boxes:
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
                print(f"[RideVision Detector] YOLOv8 inference error: {e}")

        # Generate annotated frame
        annotated_bgr = self.annotate_frame(img_bgr, detections)
        return detections, annotated_bgr

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
