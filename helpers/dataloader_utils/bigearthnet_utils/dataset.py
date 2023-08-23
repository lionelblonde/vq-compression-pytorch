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
from helpers.dataloader_utils import read_from_file, save2file
from helpers.dataloader_utils.bigearthnet_utils.constants import BAND_NAMES, RGB_BANDS_NAMES
from helpers.dataloader_utils.bigearthnet_utils.constants import LABELS, LABEL_CONVERSION
from helpers.dataloader_utils.bigearthnet_utils.constants import BANDS_10M, BANDS_20M, BAND_STATS
from helpers.dataloader_utils.bigearthnet_utils.transform_util import TransformsToolkit


def load_json(filename):
    return json.loads(Path(filename).read_text())


class BigEarthNetDataset(Dataset):

    def __init__(
        self,
        num_classes: int,
        seed: int,
        data_path: str,
        split_path: str,
        truncate_at: float,
        image_size: int = 120,
        train_stage: bool = False,
        num_transforms: int = 2,
        with_labels: bool = False,
        bands: List[str] = RGB_BANDS_NAMES,  # default: bands corresponding to RGB
        memory: bool = False,
    ):
        """`number` is the number of patches to read (-1 for all).
        `memory` indicate if the data should be first put into memory or read on the fly.
        """
        assert num_transforms in [1, 2] and isinstance(num_transforms, int)

        self.train_stage = train_stage

        if num_transforms == 2 and self.train_stage:
            self.data_augment_f = TransformsToolkit.transform_bigearthnet(image_size)
        else:
            self.data_augment_f = None

        self.num_classes = num_classes

        self.split_path = split_path
        self.with_labels = with_labels
        self.bands = bands
        self.memory = memory

        if self.train_stage:
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
        if self.train_stage:  # truncate if asked, but only in for the training set
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

        if self.with_labels:
            self.labels = np.array(self.get_labels_as_multi_hot_vector())  # always load in memory whole
            logger.info("we compute and plot the imbalance-ness now")
            n_samples, n_classes = self.labels.shape
            self.balances = self.labels.sum(axis=0) / n_samples
            # do some nice plottings in ascii style
            for c in range(n_classes):
                balance = self.balances[c]
                width = int(balance * 150)
                bar = (f"class={str(c).zfill(3)}" +
                       "[" + ("@" * width) + ("~" * (150 - width)) + "]" +
                       f"< {balance * 100:.3f}%")
                logger.info(bar)

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

    def get_labels_as_multi_hot_vector(self) -> List[List[int]]:
        """Get the true labels as a multi-hot vector"""
        where_splits = Path(self.split_path).parent.absolute()

        labels_file_path = where_splits.joinpath('labels')
        labels_file_path.mkdir(exist_ok=True)
        labels_file_path = labels_file_path.joinpath(Path(self.split_path).name)
        logger.info(f"{labels_file_path = }")

        if labels_file_path.exists():
            # read the labels from file, and return them
            content = read_from_file(labels_file_path)
            if self.train_stage:
                labels = [
                    [int(li.rstrip()) for li in line[1:-1].split(',')]
                     for i, line in enumerate(content)
                     if i in self.is2keep
                ]
            else:
                labels = [
                    [int(li.rstrip()) for li in line[1:-1].split(',')]
                     for line in content
                ]
        else:
            # create the labels, save them to file, and return them
            labels = []
            for idx, path in enumerate(tqdm(self.folder_path_list)):
                labels_raw = load_json(
                    path.joinpath(f"{self.data_point_ids[idx]}_labels_metadata.json")
                )["labels"]
                labels_instance = [0] * self.num_classes
                # for every ON label of the instance, set the index in the multi-hot ON
                for label in labels_raw:
                    assert label in LABELS['43'], f"{label} is not a valid label."
                    idx_43 = LABELS['43'].index(label)
                    if self.num_classes == 19:  # if shrinked version, use label conversion
                        for j, idx_43_group in enumerate(LABEL_CONVERSION):
                            if idx_43 in idx_43_group:
                                idx_19 = j
                                labels_instance[idx_19] = 1
                                break  # no repetition in conversion table
                        # note, some of the 43 labels are dropped completely from conversion to 19
                    else:
                        labels_instance[idx_43] = 1

                labels.append(labels_instance)
            # save to file
            save2file(labels_file_path, labels)

        return labels

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
            data = torch.Tensor(self.data[index])
        else:
            data = self.read_data([self.folder_path_list[index]], bands=bands)

        if self.data_augment_f is not None:
            output = torch.stack([
                self.data_augment_f(data),
                self.data_augment_f(data),  # two transforms
            ])
        else:
            output = data

        if self.with_labels:
            labels_for_output = torch.Tensor(self.labels[index])
            return (output, labels_for_output)
        else:
            return output

