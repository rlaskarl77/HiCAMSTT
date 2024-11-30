import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.hdc_dataset import HDC
from datasets.aicity_dataset import AiCity
from datasets.factory_dataset_cam63_72 import FactoryCam6372
from datasets.factory_dataset_cam64_73 import FactoryCam6473
from datasets.factory_dataset_cam65_74 import FactoryCam6574
from datasets.factory_dataset_cam66_76 import FactoryCam6676
from datasets.factory_dataset_cam68_78 import FactoryCam6878
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler

class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
            test_reid: bool=False,
            num_frame=100,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None
        
        self.test_reid = test_reid
        self.num_frame = num_frame

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        elif "scene" in self.dataset.lower():
            base = AiCity(self.data_dir)
        elif '14' in self.dataset.lower():
            base = FactoryCam6372(self.data_dir, self.num_frame)
        elif any(x in self.dataset.lower() for x in ['01', '06']):
            base = FactoryCam6473(self.data_dir, self.num_frame)
        elif any(x in self.dataset.lower() for x in ['02', '03', '07', '12', '13']):
            base = FactoryCam6574(self.data_dir, self.num_frame)
        elif any(x in self.dataset.lower() for x in ['10', '11']):
            base = FactoryCam6676(self.data_dir, self.num_frame)
        elif any(x in self.dataset.lower() for x in ['05', '08', '09']):
            base = FactoryCam6878(self.data_dir, self.num_frame)
        elif '20240110'or '20240415' or '20240702' in self.dataset.lower():
            base = HDC(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
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

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=1,
            num_workers=1,
        )
