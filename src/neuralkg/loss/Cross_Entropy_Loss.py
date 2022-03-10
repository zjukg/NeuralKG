import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Cross_Entropy_Loss(nn.Module):
    def __init__(self, args, model):
        super(Cross_Entropy_Loss, self).__init__()
        self.args = args
        self.model = model
        self.loss = torch.nn.BCELoss()

    def forward(self, pred, label):
        loss = self.loss(pred, label)
        return loss