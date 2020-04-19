""" A custom PyTorch dataset which can be called
by a DataLoader. """

import pathlib
from typing import Tuple
import random

import torch
import cv2


class SongDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir: pathlib.Path,
        frame_num: int = 5,
        img_width: int = 100,
        img_height: int = None,
    ) -> None:
        super().__init__()
        self.frame_num = frame_num
        self.img_width = img_width
        self.imgs = list(img_dir.glob("*.png"))
        self.len = len(self.imgs)

        # Get the bird classes
        unique_imgs = set(
            [img.stem.rsplit("-", 1)[0] for idx, img in enumerate(self.imgs)]
        )
        self.classes = {bird: idx for idx, bird in enumerate(unique_imgs)}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.imgs[idx]
        img = cv2.imread(str(img_path))
        assert img is not None, f"{img_path} can not be read."

        # Get random starting point within the image
        start_x = random.randint(0, img.shape[1] - self.frame_num * self.img_width)
        crops = [
            torch.Tensor(img[0 : img.shape[0], x : x + self.img_width])
            for x in range(start_x, start_x + self.img_width * self.frame_num, self.img_width)
        ]

        crops = torch.stack(crops).permute(0, 3, 1, 2)
        target = torch.Tensor([self.classes[img_path.stem.rsplit("-", 1)[0]]]).long()

        return crops, target

    def __len__(self) -> int:
        return self.len
