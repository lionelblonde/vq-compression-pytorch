import math

from torch.utils.data import DataLoader

from helpers.dataloader_utils.bigearthnet_utils.dataset import BigEarthNetDataset


class BigEarthNetDataloader(DataLoader):

    def __init__(self, dataset: BigEarthNetDataset, batch_size: int, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.dataset_length = len(dataset)
        if hasattr(dataset, 'balances'):
            self.balances = dataset.balances

    def __len__(self):
        """Overwrite because relays the method of the `Dataset` class otherwise"""
        if self.batch_size is None:
            raise ValueError(f"invalid batch size ({self.batch_size}); can't be None!")
        return math.ceil(self.dataset_length // self.batch_size)


def get_dataloader(
    *,
    dataset_handle: str,
    seed: str,
    data_path: str,
    split_path: str,
    batch_size: int,
    truncate_at: float,
    train_stage: bool = False,
    memory: bool = False,
    num_workers: int = 0,
    shuffle: bool = False,
):

    if dataset_handle == 'bigearthnet':
        dataloader = BigEarthNetDataloader(
            BigEarthNetDataset(
                seed=seed,
                data_path=data_path,
                split_path=split_path,
                image_size=120,
                train_stage=train_stage,
                bands=BigEarthNetDataset.all_bands(),
                memory=memory,
                truncate_at=truncate_at,
            ),
            batch_size=batch_size,
            num_workers=num_workers,  # a value of 0 plugs the memory leak (can't avoid using Python lists)
            shuffle=shuffle,
            drop_last=True,
        )
        return dataloader
    else:
        raise ValueError(f"{dataset_handle} is not a valid dataset name.")
