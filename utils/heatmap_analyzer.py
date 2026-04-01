import cv2
import numpy as np


def detect_hotspots(image_path: str) -> list[tuple[float, float]]:
    """Detect red/yellow heat blobs in a thermal/satellite image.

    Returns a list of (lat, lon) tuples for each detected hotspot centroid.
    Uses contour detection on the hot-colour mask for cleaner, fewer points.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    img = cv2.resize(img, (800, 400))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red hues wrap around 0/180 in HSV — capture both ranges
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    # Yellow-orange range
    mask3 = cv2.inRange(hsv, np.array([15, 100, 100]),  np.array([35, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    # Morphological close to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours — each contour = one hotspot
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspots: list[tuple[float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:   # skip tiny noise blobs
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        lat = 90.0 - (cy / 400.0) * 180.0
        lon = (cx / 800.0) * 360.0 - 180.0
        # Attach area as intensity proxy (normalised 0–1)
        intensity = min(1.0, area / 5000.0)
        hotspots.append((lat, lon, intensity))

    return hotspots
