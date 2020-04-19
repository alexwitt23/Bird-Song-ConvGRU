import pathlib
import random

import torch
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import cv2

import model
from data import dataset

_RUN_DIR = pathlib.Path("~/runs/bird-songs").expanduser()
_DATA_DIR = pathlib.Path("~/datasets/birdsongs-from-europe").expanduser()


if __name__ == "__main__":

    _RUN_DIR.mkdir(exist_ok=True, parents=True)

    # Create the data loaders
    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset.SongDataset(_DATA_DIR / "train"), pin_memory=True, batch_size=32
    )
    eval_loader = torch.utils.data.dataloader.DataLoader(
        dataset.SongDataset(_DATA_DIR / "eval"), pin_memory=True, batch_size=32
    )

    test_model = model.SongIdentifier(50)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(
        test_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True
    )
    test_model.cuda()

    for epoch in range(20):

        for idx, (data, target) in enumerate(train_loader):

            out = test_model(data.cuda())
            loss = loss_fn(out, target.squeeze(1).cuda())
            loss.backward()
            optimizer.step()
            if idx % 100:
                print(f"Epoch: {epoch}, Loss: {loss}")

        num_right: int = 0
        total: int = 0

        for data, target in eval_loader:

            out = test_model(data.cuda()).cpu()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            num_right += (predicted == target).sum().item()

        print(f"Epoch {epoch}, Accuracy: {num_right / total:.2}")
