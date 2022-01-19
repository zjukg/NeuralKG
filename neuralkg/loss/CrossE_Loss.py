import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed


class CrossE_Loss(nn.Module):
    def __init__(self, args, model):
        super(CrossE_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, score, label):
        return -torch.sum(torch.log(torch.clamp(score, 1e-10, 1.0)) * torch.clamp(label, 0.0, 1.0) + \
            torch.log(torch.clamp(1-score, 1e-10, 1.0)) * torch.clamp(-label, 0.0, 1.0))
