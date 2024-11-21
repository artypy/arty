import numpy as np
import torch
from os import path, mkdir
from .train import create_new
from .generate import array2angle


def cnn_predict(image: np.ndarray, kernel_size: int = 5, device=None) -> np.ndarray:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    pth = f'{path.dirname(__file__)}/saves'
    if not path.exists(pth):
        mkdir(pth)

    pth = f'{pth}/{kernel_size}.pt'
    if not path.exists(pth):
        create_new(pth, kernel_size, device)

    model = torch.load(pth, map_location='cpu').to(device)

    h, w, c = image.shape
    s, e = (kernel_size - 1) // 2, kernel_size // 2
    img = np.zeros((h + s + e, w + s + e, c))
    img[s:-e, s:-e] = image

    img = torch.tensor(img).to(device).float().T / 255

    angle = model(img).T
    shape = angle.shape[:-1]
    angle = angle.reshape(-1, 3).cpu().detach().numpy()
    angle = array2angle(angle).reshape(shape)

    return angle
