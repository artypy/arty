import numpy as np


def gradientX(image: np.ndarray) -> np.ndarray:
    image = image.sum(axis=2).astype(int)
    h, w = image.shape
    image = np.concatenate((image[:, 1:] - image[:, :-1], np.zeros((h, 1), dtype=int)), axis=1)
    return image


def gradientY(image: np.ndarray) -> np.ndarray:
    image = image.sum(axis=2).astype(int)
    h, w = image.shape
    image = np.concatenate((image[1:, :] - image[:-1, :], np.zeros((1, w), dtype=int)), axis=0)
    return image


def gradient(image: np.ndarray) -> np.ndarray:
    dx = gradientX(image)
    dy = gradientY(image)
    image = (dx ** 2 + dy ** 2) ** 0.5
    return image
