import numpy as np
import cv2


def pencil(image: np.ndarray, angles: np.ndarray, edges: np.ndarray, edge_scaler: float = 0.5, bg_scaler: float = 0.5,
           max_length: int = 7) -> np.ndarray:
    edges = edges / 765
    edges = np.clip(edges + (np.random.random(edges.shape) - 0.5) * 0.01, 0, 1)
    color = (255 - image.mean(axis=2))
    color = np.maximum(color * bg_scaler, edges ** (1 - edge_scaler) * 255)

    angles = np.radians(angles)
    dx, dy = np.cos(angles) * max_length, np.sin(angles) * max_length
    dx, dy = dx * (1 - color / 255), dy * (1 - color / 255)
    dx, dy = dx.astype(int), dy.astype(int)
    surf = np.zeros_like(image)
    priority = [(color[y, x], (x, y)) for y in range(len(edges)) for x in range(len(edges[y]))]
    priority.sort()
    for col, (x, y) in priority:
        cv2.line(surf, (x + dy[y, x], y + dx[y, x]), (x - dy[y, x], y - dx[y, x]), (col, col, col), 1)
    return 255 - surf
