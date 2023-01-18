from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from IPython import embed
from .BaseLitModel import BaseLitModel
from neuralkg.eval_task import *
from IPython import embed
from functools import partial

class CompGCNLitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        graph    = batch["graph"]  #和RGCNLitModel差别很小 形式上只有entity的区别，是否要改动
        sample   = batch["sample"]
        label    = batch["label"]
        relation = batch['relation']
        norm     = batch['norm']
        score = self.model(graph, relation, norm, sample)
        label = ((1.0 - self.args.smoothing) * label) + (
                1.0 / self.args.num_ent
        )
        loss  = self.loss(score,  label)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # pos_triple, tail_label, head_label = batch
        results = dict()
        ranks = link_predict(batch, self.model, prediction='tail')
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
        results = dict()
        ranks = link_predict(batch, self.model, prediction='tail')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    

    '''这里设置优化器和lr_scheduler'''
    def configure_optimizers(self):
        milestones = int(self.args.max_epochs / 2)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay = 1e-7)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
