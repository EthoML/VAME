import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from pathlib import Path
from typing import Optional
from vame.logging.logger import VameLogger


def save_normalization(train_dir: str | Path, data_train: np.ndarray) -> dict:
    """
    Compute the model input normalization from the training data and save it to
    seq_mean.npy and seq_std.npy in train_dir, overwriting existing files.

    Returns
    -------
    dict
        Normalization statistics, with float values under the keys ``mean`` and ``std``.
    """
    mean = float(np.mean(data_train))
    std = float(np.std(data_train))
    np.save(Path(train_dir) / "seq_mean.npy", mean)
    np.save(Path(train_dir) / "seq_std.npy", std)
    return {"mean": mean, "std": std}


def load_normalization(train_dir: str | Path) -> tuple[float, float]:
    """
    Load the model input normalization (mean, std) saved with the training data.
    Training and inference both use these values.
    """
    mean_path = Path(train_dir) / "seq_mean.npy"
    std_path = Path(train_dir) / "seq_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found in {train_dir}. Run vame.create_trainset first."
        )
    return float(np.load(mean_path)), float(np.load(std_path))


class SEQUENCE_DATASET(Dataset):
    def __init__(
        self,
        path_to_file: str,
        data: str,
        train: bool,
        temporal_window: int,
        samples_per_epoch: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Sequence Dataset.
        Normalizes the data with the statistics saved by create_trainset at:
        - project_name/
        - data/
            - train/
                - seq_mean.npy
                - seq_std.npy

        Parameters
        ----------
        path_to_file : str
            Path to the dataset files.
        data : str
            Name of the data file.
        train : bool
            Flag indicating whether it's training data.
        temporal_window : int
            Size of the temporal window.

        Returns
        -------
        None
        """
        self.logger_config = kwargs.get("logger_config", VameLogger(__name__))
        self.logger = self.logger_config.logger

        self.temporal_window = temporal_window
        self.X = np.load(path_to_file + data)
        if self.X.shape[0] > self.X.shape[1]:
            self.X = self.X.T

        self.data_points = len(self.X[0, :])

        self.mean, self.std = load_normalization(path_to_file)

        # Normalize once and store as float32. Previously each __getitem__ did
        # (x - mean) / std in float64 and the train loop cast to float32 — same
        # numbers, but doing it once here halves host->device bandwidth and drops
        # a per-batch device cast.
        self.X = ((self.X - self.mean) / self.std).astype(np.float32)

        # Number of random-crop samples drawn per epoch. None => one per frame
        # (legacy behavior); set to decouple epoch length from dataset size.
        self.samples_per_epoch = samples_per_epoch

        if train:
            self.logger.info("Initialize train data. Datapoints %d" % self.data_points)
        else:
            self.logger.info("Initialize test data. Datapoints %d" % self.data_points)

    def __len__(self) -> int:
        """
        Return the number of data points.

        Returns
        -------
        int
            Number of data points.
        """
        return self.samples_per_epoch or self.data_points

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a normalized sequence at the specified index.

        Parameters
        ----------
        index : int
            Index of the item.

        Returns
        -------
        torch.Tensor
            Normalized sequence data at the specified index.
        """
        temp_window = self.temporal_window
        nf = self.data_points
        start = np.random.choice(nf - temp_window)
        end = start + temp_window
        # self.X is already normalized + float32; copy so the slice is contiguous
        # and not a view shared across worker processes.
        return torch.from_numpy(self.X[:, start:end].copy())
