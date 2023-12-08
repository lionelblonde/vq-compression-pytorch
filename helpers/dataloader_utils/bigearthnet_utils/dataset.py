import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from helpers import logger
from helpers.dataloader_utils import read_from_file
from helpers.dataloader_utils.bigearthnet_utils.constants import BAND_NAMES, RGB_BANDS_NAMES
from helpers.dataloader_utils.bigearthnet_utils.constants import LABELS, LABEL_CONVERSION
from helpers.dataloader_utils.bigearthnet_utils.constants import BANDS_10M, BANDS_20M, BAND_STATS


def load_json(filename):
    return json.loads(Path(filename).read_text())


class BigEarthNetDataset(Dataset):

    def __init__(
        self,
        seed: int,
        data_path: str,
        split_path: str,
        truncate_at: float,
        image_size: int = 120,
        train_stage: bool = False,
        bands: List[str] = RGB_BANDS_NAMES,  # default: bands corresponding to RGB
        memory: bool = False,
    ):
        """`number` is the number of patches to read (-1 for all).
        `memory` indicate if the data should be first put into memory or read on the fly.
        """

        self.split_path = split_path
        self.bands = bands
        self.memory = memory

        if train_stage:
            assert 0 < truncate_at <= 100
            tot_len = len(read_from_file(self.split_path, data_path))
            self.truncate_at = int(
                (truncate_at / 100.) * tot_len
            )  # transform the % to keep into the # of samples to keep
            logger.info(
                f"#samplesKEPT={self.truncate_at} -|-"
                f"OVER TOT={tot_len} -|-"
                f"i.e. PERC={int(self.truncate_at) / tot_len * 100.}%"
            )  # sanity check

        content = read_from_file(self.split_path, parent=data_path)
        if train_stage:  # truncate if asked, but only in for the training set
            all_the_is = np.arange(0, len(content))
            self.is2keep = np.random.default_rng(seed).choice(
                all_the_is,
                self.truncate_at,
                replace=False,  # no repetitions
            )  # make it an attribute since used later
            content = [Path(line) for i, line in enumerate(content) if i in self.is2keep]
        else:
            content = [Path(line) for line in content]  # keep everything for val and test
        self.folder_path_list = np.array(content)

        self.verify_bands(self.bands)

        self.data_point_ids = np.array(self.get_data_point_ids())

        if self.memory:
            self.data = self.read_data(self.folder_path_list)

    @staticmethod
    def rgb() -> List[str]:
        return RGB_BANDS_NAMES

    @staticmethod
    def all_bands() -> List[str]:
        return BANDS_10M + BANDS_20M  # BANDS_60M are ignored (TUB does exactly this)

    def verify_bands(self, bands: List[str]) -> None:
        for band in bands:
            if band not in BAND_NAMES:
                raise ValueError(f"{band} is not a valid band.")

    def get_data_point_ids(self) -> List[str]:
        return [path.name for path in self.folder_path_list]

    def read_data(
        self, folder_path_list: List[Path], bands: Optional[List[str]] = None
    ) -> torch.Tensor:
        if bands is None:
            bands = self.bands
        data = []
        for path in folder_path_list:
            name = path.name
            bands_data = []
            for band_name in bands:
                band_path = path.joinpath(f"{name}_{band_name}.tif")
                with rasterio.open(band_path, driver="GTiff", sharing=False) as band_file:
                    band_data = band_file.read(1)  # open the tif image as a numpy array
                    # Resize depending on the resolution
                    if band_name in BANDS_20M:
                        # Carry out a bicubic interpolation (TUB does exactly this)
                        band_data = cv2.resize(band_data, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
                    # We have already ignored the 60M ones, and we keep the 10M ones intact

                    # Normalize using the stats provided by TUB
                    band_data = ((band_data - BAND_STATS['mean'][band_name]) /
                                 BAND_STATS['std'][band_name]).astype(np.float32)

                    bands_data.append(band_data)

                band_file.close()

            bands_data = np.stack(bands_data)

            data.append(bands_data)
        data_np = np.array(data)
        del data
        if len(data_np) == 1:
            return torch.Tensor(data_np[0])
        else:
            return torch.Tensor(data_np)

    def __len__(self) -> int:
        """Returns number of instances in dataset."""
        return len(self.data_point_ids)

    def __getitem__(
        self, index: int, bands: Optional[List[str]] = None
    ) -> Union[Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor], Union[torch.Tensor, List[torch.Tensor]]]:
        assert isinstance(index, int), "index must be int"

        if self.memory:
            output = torch.Tensor(self.data[index])
        else:
            output = self.read_data([self.folder_path_list[index]], bands=bands)

        return output

