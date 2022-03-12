import argparse
import pytorch_lightning as pl
import torch
from collections import defaultdict as ddict

from neuralkg import loss
import numpy as np



class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = args
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model)


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    
    def collect_results(self, results, mode):
        """Summarize the results of each batch and calculate the final result of the epoch
        Args:
            results ([type]): The results of each batch
            mode ([type]): Eval or Test
        Returns:
            dict: The final result of the epoch
        """
        outputs = ddict(float)
        count = np.array([o["count"] for o in results]).sum()
        for metric in list(results[0].keys())[1:]:
            final_metric = "|".join([mode, metric])
            outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count, decimals=3).item()
        return outputs
    
    