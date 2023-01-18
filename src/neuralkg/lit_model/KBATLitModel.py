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

class KBATLitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        num_epoch  = self.current_epoch
        if num_epoch < self.args.epoch_GAT: 
            model   = "GAT"
            adj     = batch['adj_matrix']
            n_hop   = batch['n_hop']
            pos_triple = batch['triples_GAT_pos']
            neg_triple = batch['triples_GAT_neg']
            pos_score  = self.model(pos_triple, model, adj, n_hop)
            neg_score  = self.model(neg_triple, model, adj, n_hop)
            loss       = self.loss(model, pos_score, neg_score)
        else:
            model   = "ConvKB"
            triples = batch['triples_Con']
            label   = batch['label']
            score   = self.model(triples, model)
            loss    = self.loss(model, score, label=label)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss

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
        if self.current_epoch < self.args.epoch_GAT:
            optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=1e-6)
            StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)

        else:
            optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=1e-5)
            StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)

        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
