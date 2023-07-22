import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from helpers.dataloader_utils import read_from_file, read_path_from_file, read_list_from_file, save2file
from helpers.dataloader_utils.bigearthnet_utils.constants import BAND_NAMES, RGB_BANDS_NAMES, LABELS
from helpers.dataloader_utils.bigearthnet_utils.constants import BANDS_10M, BANDS_20M, BAND_STATS
from helpers.dataloader_utils.bigearthnet_utils.transform_util import TransformsToolkit


def load_json(filename):
    return json.loads(Path(filename).read_text())


class BigEarthNetDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        split_path: str,
        image_size: int = 120,
        train_stage: bool = False,
        num_transforms: int = 2,
        with_labels: bool = False,
        bands: List[str] = RGB_BANDS_NAMES,  # default: bands corresponding to RGB
        memory: bool = False,
        truncate_at: int = 100,
    ):
        """`number` is the number of patches to read (-1 for all).
        `memory` indicate if the data should be first put into memory or read on the fly.
        """
        assert num_transforms in [1, 2] and isinstance(num_transforms, int)

        if num_transforms == 1:
            self.transform = None
        else:  # num_transforms == 2
            # We have checked above that only one of the condition is true
            if train_stage:
                self.transform = self.train_transform(image_size)
            else:
                self.transform = self.eval_transform(image_size)

        self.split_path = split_path
        self.with_labels = with_labels
        self.bands = bands
        self.memory = memory

        assert 0 < truncate_at <= 100
        tot_len = len(read_from_file(self.split_path, data_path))
        self.truncate_at = int((truncate_at / 100.) * tot_len)  # transform the % to keep into the # of samples to keep
        print(self.truncate_at, tot_len, f"{int(self.truncate_at) / tot_len * 100}%")  # sanity check

        self.verify_bands(self.bands)

        self.folder_path_list = np.array(read_path_from_file(self.split_path, data_path, self.truncate_at))
        self.data_point_ids = np.array(self.get_data_point_ids())

        if self.memory:
            self.data = self.read_data(self.folder_path_list)

        if self.with_labels:
            self.labels = np.array(self.get_labels_as_multi_hot_vector())  # always load in memory whole

    @staticmethod
    def rgb() -> List[str]:
        return RGB_BANDS_NAMES

    @staticmethod
    def all_bands() -> List[str]:
        return BANDS_10M + BANDS_20M  # BANDS_60M are ignored (TUB does exactly this)

    @staticmethod
    def num_classes() -> int:
        return len(LABELS)

    def train_transform(self, image_size: int) -> transforms.Compose:
        return TransformsToolkit.transform_bigearthnet_train(image_size)

    def eval_transform(self, image_size: int) -> transforms.Compose:
        return TransformsToolkit.transform_bigearthnet_eval(image_size)

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
        print("1", "labels from", labels_file_path)

        if labels_file_path.exists():
            # read the labels from file, and return them
            labels = read_list_from_file(labels_file_path, truncate_at=self.truncate_at)
            print("2", "# of labels", len(labels))

        else:
            # create the labels, save them to file, and return them
            labels = []
            for idx, path in enumerate(tqdm(self.folder_path_list)):
                labels_raw = load_json(
                    path.joinpath(f"{self.data_point_ids[idx]}_labels_metadata.json")
                )["labels"]
                labels_instance = [0] * self.num_classes()
                # for every ON label of the instance, set the index in the multi-hot ON
                for label in labels_raw:
                    assert label in LABELS, f"{label} is not a valid label."
                    idx = LABELS.index(label)

                    labels_instance[idx] = 1
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

            # normalize the bands
            # bands_data = self.normalize_bands(bands_data)  # between 0 and 255
            # bands_data = self.normalize_bands(bands_data, desired_max=1)  # between 0 and 1

            data.append(bands_data)
        data_np = np.array(data)
        del data
        if len(data_np) == 1:
            return torch.Tensor(data_np[0])
        else:
            return torch.Tensor(data_np)

    # def normalize_bands(
    #     self,
    #     bands_data: np.ndarray,
    #     desired_max: int = 255,
    #     band_min: int = 0,
    #     band_max: int = 10000,
    #     normalize_per_band: bool = True,
    # ) -> np.ndarray:
    #     bands_data = bands_data.astype(float)
    #     if normalize_per_band:
    #         for band_idx in range(bands_data.shape[0]):
    #             np.clip(bands_data[band_idx], band_min, band_max)
    #             bands_data[band_idx] = (
    #                 bands_data[band_idx] / bands_data[band_idx].max()
    #             ) * desired_max

    #     else:
    #         np.clip(bands_data, band_min, band_max)
    #         bands_data = (bands_data / bands_data.max()) * desired_max
    #     return bands_data

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

        if self.transform is not None:
            output = torch.stack([
                self.transform(torch.Tensor(data)),
                self.transform(torch.Tensor(data)),  # two transforms
            ])
        else:
            output = data

        if self.with_labels:
            labels_for_output = torch.Tensor(self.labels[index])
            return (output, labels_for_output)
        else:
            return output

