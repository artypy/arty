from torch import nn
from .generate import get_batch
import torch


def create_new(path, kernel_size, device):
    model = nn.Sequential(
        nn.Conv2d(3, kernel_size ** 2, kernel_size),
        nn.ReLU(),
        nn.Conv2d(kernel_size ** 2, 3, 1),
        nn.Sigmoid(),
    ).to(device)

    print('training...')

    train(model, kernel_size, device, path)


def train(model, kernel_size, device, save):
    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best = 1
    remaining = 1000

    while remaining:
        input, output = get_batch(kernel_size)
        input = torch.tensor(input).to(device).permute(0, 3, 2, 1).float()
        output = torch.tensor(output).to(device).float()

        out = model(input)
        loss = loss_fn(out, output[:, :, None, None])

        remaining -= 1
        if loss < best:
            remaining += 10
            best = loss
            torch.save(model, save)

        optim.zero_grad()
        loss.backward()
        optim.step()
