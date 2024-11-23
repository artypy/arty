import numpy as np
from queue import PriorityQueue


def conditional(angles: np.ndarray, edges: np.ndarray) -> np.ndarray:
    def closest(orig, angle):
        angles = [(abs(orig - i), i) for i in [angle - 180, angle, angle + 180]]
        return min(angles)[1]

    h, w = angles.shape
    edges = edges.copy()
    used = np.zeros_like(angles)
    q = PriorityQueue()
    for y in range(len(edges)):
        for x in range(len(edges[y])):
            q.put((-edges[y, x], (x, y)))
    while not q.empty():
        _, (x, y) = q.get()
        if used[y, x]: continue
        used[y, x] = 1
        edge = edges[y, x]
        angle = angles[y, x]
        for X in range(max(0, x - 1), min(x + 2, w)):
            for Y in range(max(0, y - 1), min(y + 2, h)):
                if used[Y, X]: continue
                ang = closest(angle, angles[Y, X])
                ed = edges[Y, X]
                ang = (ang + (angle - ang) * (edge - ed) / max(1, edge)) % 180
                angles[Y, X] = ang
                edges[Y, X] = (edge + ed) / 2
                q.put((-edges[Y, X], (X, Y)))
    return angles
