import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.stream_dataset import LoadStreams
from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.wildtrack_dataset_3cam import Wildtrack3cam
from datasets.hdc_dataset import HDC
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler


class StreamDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            sources: str = "sources.txt",
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sources = sources
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            if '3cam' in self.dataset.lower():
                base = Wildtrack3cam(self.data_dir)
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        elif '20240110'or '20240415' in self.dataset.lower():
            base = HDC(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        if stage == 'predict':
            self.data_predict = LoadStreams(
                base,
                sources=self.sources,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def predict_dataloader(self):
        return self.data_predict
