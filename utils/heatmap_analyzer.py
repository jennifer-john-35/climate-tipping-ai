import cv2
import numpy as np


def detect_hotspots(image):

    img = cv2.imread(image)

    img = cv2.resize(img, (800, 400))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red/yellow high-risk colors
    lower = np.array([0, 120, 120])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    coords = np.column_stack(np.where(mask > 0))

    hotspots = []

    for y, x in coords[::200]:
        lat = 90 - (y/400)*180
        lon = (x/800)*360 - 180

        hotspots.append((lat, lon))

    return hotspots
