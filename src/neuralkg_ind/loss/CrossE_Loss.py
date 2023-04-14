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
        pos = torch.log(torch.clamp(score, 1e-10, 1.0)) * torch.clamp(label, 0.0, 1.0)
        neg = torch.log(torch.clamp(1-score, 1e-10, 1.0)) * torch.clamp(-label, 0.0, 1.0)
        num_pos = torch.sum(torch.clamp(label, 0.0, 1.0), -1)
        num_neg = torch.sum(torch.clamp(-label, 0.0, 1.0), -1)
        loss = - torch.sum(torch.sum(pos, -1)/num_pos) - torch.sum(torch.sum(neg, -1)/num_neg)
        return loss
