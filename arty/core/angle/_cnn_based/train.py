from torch import nn
from .generate import get_batch
import torch


def create_new(path, kernel_size, device):
    model = nn.Sequential(
        nn.Conv2d(3, kernel_size ** 2, kernel_size, 1, kernel_size // 2),
        nn.Sigmoid(),
        nn.Conv2d(kernel_size ** 2, 3, 1, 1),
        nn.Sigmoid(),
    ).to(device)

    train(model, kernel_size, device, path)
    print('saved')


def train(model, kernel_size, device, save):
    h = kernel_size // 2

    optim = torch.optim.Adam(model.parameters(), 3e-4)
    loss_fn = nn.MSELoss()

    best = 1
    remaining = 5000
    while remaining:
        input, output = get_batch(kernel_size)
        input = torch.tensor(input).to(device).permute(0, 3, 2, 1).float()
        output = torch.tensor(output).to(device).float()

        out = model(input)
        res = out.clone()
        res[:, :, h, h] = output
        loss = loss_fn(out, res)
        print(loss)

        remaining -= 1
        if loss < best:
            print('saved')
            remaining += 5
            best = loss
            torch.save(model, save)

        optim.zero_grad()
        loss.backward()
        optim.step()
