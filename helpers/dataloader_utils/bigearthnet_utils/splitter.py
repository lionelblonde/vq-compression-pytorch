import random
from pathlib import Path, PurePath
from typing import Dict, Optional

from helpers import logger
from helpers.dataloader_utils import path2str, save2file


def split(
    data_path: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: Optional[int] = None,  # semantically, important to leave the seed as parameter
    dataset_name: Optional[str] = None,
    truncate_at=None,
) -> Dict[str, str]:

    if seed is None:
        raise ValueError("at this stage, seed must not be None")
    random.seed(seed)  # already done in main, but can't hurt here

    if dataset_name is None:
        raise ValueError("one must give a dataset name")
    dataset_split_folder = (f"./splits/split-{dataset_name}"
                            f"-val{val_split}"
                            f"-test{test_split}"
                            f"-seed{str(seed).zfill(2)}")

    if truncate_at is not None:
        # add the number of data points we truncate at
        dataset_split_folder += f"-truncate_at{truncate_at}"

    # Make it a proper Path object
    dataset_split_folder = Path(dataset_split_folder)

    if dataset_split_folder.exists():
        # if the folder already exists, use the pre-existing one!
        logger.info("dataset split folder already exist. moving on")
        paths_dict = {
            "train": path2str(dataset_split_folder.joinpath("train.txt")),
            "val": path2str(dataset_split_folder.joinpath("val.txt")),
            "test": path2str(dataset_split_folder.joinpath("test.txt")),
        }
        return paths_dict

    dataset_split_folder.mkdir(exist_ok=True, parents=True)

    logger.info("dataset split folders O.K.")

    dataset_folder = Path(data_path)
    folder_path_list = []
    if truncate_at is not None:
        i = 0
        for path in dataset_folder.iterdir():
            if path.is_dir():
                if i >= truncate_at:
                    break
                else:
                    folder_path_list.append(PurePath(path2str(path)).name)  # add only the child dir name
                    i += 1
    else:
        folder_path_list = [PurePath(path2str(path)).name for path in dataset_folder.iterdir() if path.is_dir()]

    random.shuffle(folder_path_list)  # shuffle the order of paths to data points

    train_split = max(1 - val_split - test_split, 0)
    n_train = int(train_split * len(folder_path_list))

    if test_split == 0.0:
        n_val = len(folder_path_list)
    else:
        n_val = n_train + int(val_split * len(folder_path_list))

    train = folder_path_list[:n_train]
    val = folder_path_list[n_train:n_val]
    test = folder_path_list[n_val:]
    assert len(train) + len(val) + len(test) == len(folder_path_list), (
        "splitted folders should have same number of files:\n"
        f"len(train)={len(train)}, len(val)={len(val)}, len(test)={len(test)}\n"
        f"len(total)={len(folder_path_list)}"
    )

    save2file(dataset_split_folder.joinpath("train.txt"), train)
    save2file(dataset_split_folder.joinpath("val.txt"), val)
    save2file(dataset_split_folder.joinpath("test.txt"), test)

    paths_dict = {
        "train": path2str(dataset_split_folder.joinpath("train.txt")),
        "val": path2str(dataset_split_folder.joinpath("val.txt")),
        "test": path2str(dataset_split_folder.joinpath("test.txt")),
    }

    return paths_dict


def split_datasets(
    data_path: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: Optional[int] = None,
    truncate_at=None,
) -> Dict[str, str]:

    paths_dict = {}

    if "BigEarthNet-v1.0" in data_path:
        paths_dict = split(
            data_path=data_path,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            dataset_name="BigEarthNet-v1.0",
            truncate_at=truncate_at,
        )

    return paths_dict
