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

import pickle
import time

from functools import partial

class IterELitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.epoch=0
    

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
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
    
    def training_epoch_end(self, results):
        self.epoch+=1
        if self.epoch % self.args.update_axiom_per == 0 and self.epoch !=0:
                # axioms include probability for each axiom in axiom pool
                # order: ref, sym, tran, inver, sub, equi, inferC
                # update_axioms:
                #            1) calculate probability for each axiom in axiom pool with current embeddings
                #            2) update the valid_axioms
                axioms_probability = self.update_axiom()
                updated_train_data = self.model.update_train_triples(epoch = self.epoch, update_per= self.args.update_axiom_per)
                if updated_train_data:
                    self.trainer.datamodule.data_train=updated_train_data
                    self.trainer.datamodule.train_sampler.count = self.trainer.datamodule.train_sampler.count_frequency(updated_train_data)


    def update_axiom(self):
        time_s = time.time()
        axiom_pro = self.model.run_axiom_probability()
        time_e = time.time()
        print('calculate axiom score:', time_e -time_s)
        with open('./save_axiom_prob/axiom_prob.pickle', 'wb') as f: pickle.dump(axiom_pro, f, pickle.HIGHEST_PROTOCOL)
        with open('./save_axiom_prob/axiom_pools.pickle', 'wb') as f: pickle.dump(self.model.axiompool, f, pickle.HIGHEST_PROTOCOL)
        self.model.update_valid_axioms(axiom_pro)
        return self.model.run_axiom_probability()

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
        milestones = [5,50]
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
