#!/user/bin/env python3
""" A script to convert .mp3 files into spectrograms. 
The dataset was downloaded:
from https://www.kaggle.com/monogenea/birdsongs-from-europe """

import pathlib
import random

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import tqdm
import multiprocessing

_DATA_DIR = pathlib.Path("~/datasets/birdsongs-from-europe").expanduser()
_SAVE_DIR = _DATA_DIR / "specs"


def process_data(
    mp3_dir: pathlib.Path, train_dir: pathlib.Path, eval_dir: pathlib.Path
) -> None:
    """ Convert the mp3 files to spectrogram .pngs. 
    Args:
        mp3_dir: directory of mp3s.
        train_dir: Directory to save training spectrograms to.
        eval_dir: Directory to save evaluation spectrograms to.
    """

    for mp3 in tqdm.tqdm((mp3_dir).glob("*mp3")):

        y, sr = librosa.load(mp3)
        window_size = 2048

        window = np.hanning(window_size)
        stft = librosa.core.spectrum.stft(
            y, n_fft=window_size, hop_length=2048, window=window
        )
        out = 2 * np.abs(stft) / np.sum(window)

        # For plotting headlessly
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(
            librosa.amplitude_to_db(out, ref=np.max),
            ax=ax,
            y_axis="mel",
            x_axis="time",
            hop_length=window_size,
        )
        ax.axis("off")
        if random.randint(0, 100) <= 20:
            fig.savefig(eval_dir / f"{mp3.stem}.png")
        else:
            fig.savefig(train_dir / f"{mp3.stem}.png")
        plt.close()


if __name__ == "__main__":

    random.seed(42)

    assert _DATA_DIR.is_dir(), f"{_DATA_DIR} can not be found."

    train_dir = _DATA_DIR / "train"
    eval_dir = _DATA_DIR / "eval"
    train_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    process_data(_DATA_DIR / "mp3", train_dir, eval_dir)
