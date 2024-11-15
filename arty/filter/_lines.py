from arty import core
import numpy as np

def lines(image: np.ndarray, to_blur: bool = False, to_noise: bool = False) -> np.ndarray:
    if to_blur:
        image = core.filter.blur.gauss(image)

    if to_noise:
        image = core.filter.noise(image)

    dx, dy = core.edge.gradientX(image), core.edge.gradientY(image)
    edges = core.edge.gradient(image)
    angles = core.angle.gradient_truth(dx, dy)
    lines = core.filter.lines(image, angles, edges)

    return lines