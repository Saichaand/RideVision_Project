"""
Generates realistic sample road images with potholes for test and demo purposes.
"""
import os
import numpy as np
import cv2

SAMPLES_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(SAMPLES_DIR, exist_ok=True)


def create_road_sample(filename: str, pothole_type: str = "severe"):
    width, height = 640, 480
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Asphalt base texture with noise
    road_gray = np.random.normal(70, 8, (height, width)).astype(np.uint8)
    img[:, :, 0] = road_gray
    img[:, :, 1] = road_gray
    img[:, :, 2] = road_gray

    # Perspective road lane markings (white/yellow dashes)
    pts_left = np.array([[240, 100], [250, 100], [80, 480], [60, 480]], np.int32)
    pts_right = np.array([[390, 100], [400, 100], [580, 480], [560, 480]], np.int32)
    cv2.fillPoly(img, [pts_left], (220, 220, 220))
    cv2.fillPoly(img, [pts_right], (220, 220, 220))

    # Center dashed yellow line
    for y in range(120, 480, 70):
        w_factor = 2 + int((y / 480) * 8)
        cv2.rectangle(img, (320 - w_factor // 2, y), (320 + w_factor // 2, y + 40), (30, 200, 240), -1)

    if pothole_type == "severe":
        # Dark, irregular cavity
        center = (280, 320)
        axes = (85, 55)
        angle = -15
        cv2.ellipse(img, center, axes, angle, 0, 360, (25, 25, 28), -1)
        # Inner depth shadow
        cv2.ellipse(img, (center[0] - 5, center[1] + 5), (int(axes[0]*0.75), int(axes[1]*0.65)), angle, 0, 360, (12, 12, 15), -1)
        # Rough jagged edges
        for a in range(0, 360, 15):
            rad = np.radians(a)
            r = axes[0] + np.random.randint(-12, 12)
            px = int(center[0] + r * np.cos(rad) * 0.9)
            py = int(center[1] + r * np.sin(rad) * 0.6)
            cv2.circle(img, (px, py), np.random.randint(6, 14), (20, 20, 22), -1)
        # Cracked asphalt texture around rim
        cv2.ellipse(img, center, (axes[0] + 18, axes[1] + 12), angle, 0, 360, (40, 40, 45), 2)

    elif pothole_type == "moderate":
        center = (360, 310)
        axes = (50, 32)
        angle = 10
        cv2.ellipse(img, center, axes, angle, 0, 360, (28, 28, 30), -1)
        cv2.ellipse(img, (center[0] - 3, center[1] + 3), (int(axes[0]*0.7), int(axes[1]*0.6)), angle, 0, 360, (15, 15, 18), -1)
        cv2.ellipse(img, center, (axes[0] + 10, axes[1] + 8), angle, 0, 360, (45, 45, 50), 2)

    elif pothole_type == "clean":
        # Smooth clean road surface, no potholes
        pass

    out_path = os.path.join(SAMPLES_DIR, filename)
    cv2.imwrite(out_path, img)
    print(f"Created sample image: {out_path}")


if __name__ == "__main__":
    create_road_sample("sample_severe_pothole.jpg", "severe")
    create_road_sample("sample_moderate_pothole.jpg", "moderate")
    create_road_sample("sample_clean_road.jpg", "clean")
