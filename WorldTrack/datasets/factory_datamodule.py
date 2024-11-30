import os
from typing import Optional, List

import lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.hdc_dataset import HDC
from datasets.aicity_dataset import AiCity
from datasets.factory_dataset_cam63_72 import FactoryCam6372
from datasets.factory_dataset_cam64_73 import FactoryCam6473
from datasets.factory_dataset_cam65_74 import FactoryCam6574
from datasets.factory_dataset_cam66_76 import FactoryCam6676
from datasets.factory_dataset_cam68_78 import FactoryCam6878
from datasets.sampler import TemporalSampler
from datasets.pedestrian_dataset import PedestrianDataset

class FactoryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dirs: list = ["../data/MultiviewX"],
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
            test_reid: bool = False,
            num_frames: list = [100],
            test_scene: Optional[str] = None,
            test_scene_num_frame: Optional[int] = None,
    ):
        super().__init__()
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        self.num_frames = [num_frames] if isinstance(num_frames, int) else num_frames
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

        self.test_reid = test_reid
        self.batch_size = batch_size
        self.test_scene = test_scene if test_scene is not None else self.data_dirs[0]
        self.test_scene_num_frame = test_scene_num_frame if test_scene is not None else self.num_frames[0]
        self.groups = {
            "6372": ['14'],
            "6473": ['01', '06'],
            "6574": ['02', '03', '07', '12', '13'],
            "6676": ['10', '11'],
            "6878": ['05', '08', '09'],
        }

    def _validate_groups(self):
        group_keys = list(self.groups.keys())
        matched_group = None

        for data_dir in self.data_dirs:
            found = False
            for group, identifiers in self.groups.items():
                if any(x in data_dir.lower() for x in identifiers):
                    if matched_group is None:
                        matched_group = group
                    elif matched_group != group:
                        raise ValueError(
                            f"Data directories belong to multiple groups. "
                            f"Found mismatch: {data_dir} does not belong to group {matched_group}."
                        )
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Unknown dataset type in path: {data_dir}. It does not match any group."
                )

        if matched_group is None:
            raise ValueError("No valid group matched for the provided data_dirs.")

        return matched_group

    def _get_base(self, group_key, data_dirs, num_frames):
        if group_key == "6372":
            return FactoryCam6372(data_dirs, num_frames)
        elif group_key == "6473":
            return FactoryCam6473(data_dirs, num_frames)
        elif group_key == "6574":
            return FactoryCam6574(data_dirs, num_frames)
        elif group_key == "6676":
            return FactoryCam6676(data_dirs, num_frames)
        elif group_key == "6878":
            return FactoryCam6878(data_dirs, num_frames)
        else:
            raise ValueError(f"Unexpected error with group matching: {group_key}")

    def setup(self, stage: Optional[str] = None):
        group_key = self._validate_groups()

        if stage == 'fit':
            datasets = []
            for data_dir, num_frame in zip(self.data_dirs, self.num_frames):
                base = self._get_base(group_key, data_dir, num_frame)
                dataset = PedestrianDataset(
                        base,
                        is_train=True,
                        resolution=self.resolution,
                        bounds=self.bounds,
                )
                datasets.append(dataset)
            self.data_train = ConcatDataset(datasets)

        if stage == 'fit' or stage == 'validate':
            datasets = []
            for data_dir, num_frame in zip(self.data_dirs, self.num_frames):
                base = self._get_base(group_key, data_dir, num_frame)
                dataset = PedestrianDataset(
                        base,
                        is_train=False,
                        resolution=self.resolution,
                        bounds=self.bounds,
                )
                datasets.append(dataset)
            self.data_val = ConcatDataset(datasets)

        if stage == 'test':
            base = self._get_base(group_key, self.test_scene, self.test_scene_num_frame)
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
                test_reid=self.test_reid
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )
