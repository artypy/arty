import numpy as np
from ._points import points
from ..edge import gradient
import cv2


def strokes(image: np.ndarray, angles: np.ndarray, mx_length: int = 10) -> np.ndarray:
    pts = points(image, 7, 1, 0)
    grad = gradient(image) / 765
    surf = np.ones_like(image) * 255
    image = 1 - image.mean(axis=2) / 255
    h, w = pts.shape

    angles[grad < 0.05] = np.random.random(angles.shape)[grad < 0.05] * 180
    angles = np.radians(angles)
    X, Y = np.cos(angles), np.sin(angles)
    for y in range(h):
        for x in range(w):
            if pts[y, x]: continue
            ln = image[y, x] * mx_length / 2
            dx, dy = X[y, x], Y[y, x]
            cv2.line(surf, (int(x + ln * dy), int(y + ln * dx)), (int(x - ln * dy), int(y - ln * dx)), (50, 50, 50), 1)
    return surf
