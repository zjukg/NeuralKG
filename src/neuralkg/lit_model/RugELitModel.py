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
from neuralkg import loss
from .BaseLitModel import BaseLitModel
from neuralkg.eval_task import *
from IPython import embed

from functools import partial
from neuralkg.data import RuleDataLoader
from tqdm import tqdm
import pdb


class RugELitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.args = args
        self.temp_list = []
        self.rule_dataloader = RuleDataLoader(self.args)
        tq = tqdm(self.rule_dataloader, desc='{}'.format('rule'), ncols=0)
        print('start first load')
        for new_data in tq:
            self.temp_list.append(new_data)
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, mode)
        rule, confidence, triple_num = self.temp_list[0][0], self.temp_list[0][1], self.temp_list[0][2]
        loss = self.loss(pos_score, neg_score, rule, confidence, triple_num, len(pos_sample))
        self.temp_list.remove(self.temp_list[0])
        self.log("Train|loss", loss, on_step=False, on_epoch=True)
        return loss 

    def training_epoch_end(self, training_step_outputs):

        self.temp_list = []
        print('start reload')
        tq = tqdm(self.rule_dataloader, desc='{}'.format('rule'), ncols=0)
        for new_data in tq:
            self.temp_list.append(new_data)

            
    def validation_step(self, batch, batch_idx):
        # pos_triple, tail_label, head_label = batch
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

    '''这里设置优化器和lr_scheduler'''

    def configure_optimizers(self):
        # milestones = int(self.args.max_epochs / 2)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer}
        return optim_dict
