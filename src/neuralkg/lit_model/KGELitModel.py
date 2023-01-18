from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from .BaseLitModel import BaseLitModel
from IPython import embed
from neuralkg.eval_task import *
from IPython import embed

from functools import partial

class KGELitModel(BaseLitModel):
    """Processing of training, evaluation and testing.
    """

    def __init__(self, model, args):
        super().__init__(model, args)

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser
    
    def training_step(self, batch, batch_idx):
        """Getting samples and training in KG model.
        
        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, mode)
        if self.args.use_weight:
            subsampling_weight = batch["subsampling_weight"]
            loss = self.loss(pos_score, neg_score, subsampling_weight)
        else:
            loss = self.loss(pos_score, neg_score)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Getting samples and test in KG model.
        
        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
     

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        milestones = int(self.args.max_epochs / 2)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
