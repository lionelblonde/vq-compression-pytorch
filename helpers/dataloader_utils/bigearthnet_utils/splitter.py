from pathlib import Path
from typing import Dict

from helpers.dataloader_utils import path2str


def split_bigearthnet_official() -> Dict[str, str]:
    where_splits = ("./splits/BigEarthNet-S2_43-classes_OFFICIAL/splits")
    txt_files = ['train.txt', 'val.txt', 'test.txt']
    return [path2str(Path(where_splits).joinpath(f)) for f in txt_files]


def split_datasets(dataset_handle: str = None) -> Dict[str, str]:
    paths_list = []

    if dataset_handle == "BigEarthNet-v1.0":
        paths_list = split_bigearthnet_official()

    return paths_list
