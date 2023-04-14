"""Base DataModule class."""
from pathlib import Path
from typing import Dict
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


BATCH_SIZE = 8
NUM_WORKERS = 8


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args


    @staticmethod
    def add_to_argparse(parser):
        
        parser.add_argument(
            "--train_bs",
            type=int,
            default=0,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_batches",
            type=int,
            default=0,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--eval_bs",
            type=int,
            default=16,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default="./dataset/WN18RR",
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--test_bs",
            type=int,
            default=None,
            help="Number of examples to operate on per forward step.",
        )
        return parser

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_config(self):
        return dict(num_labels=self.num_labels)