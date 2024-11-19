import numpy as np
import torch
from os import path
from .train import create_new
from .generate import array2angle


def cnn_predict(image: np.ndarray, kernel_size: int = 5, device=None) -> np.ndarray:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    pth = f'{path.dirname(__file__)}/saves/{kernel_size}.pt'
    if not path.exists(pth):
        create_new(pth, kernel_size, device)
    model = torch.load(pth, map_location='cpu').to(device)
    print(image.shape)
    img = torch.tensor(image).to(device).float().T / 255
    angle = model(img).T.reshape(-1, 3).cpu().detach().numpy()

    #angle = model(img).T
    #shape = angle.shape[:-1]
    #angle = angle.reshape(-1, 3).cpu().detach().numpy()
    #angle = array2angle(angle).reshape(shape)

    angle = array2angle(angle).reshape(image.shape[:2])
    return angle