import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Margin_Loss(nn.Module):
    def __init__(self, args, model):
        super(Margin_Loss, self).__init__()
        self.args = args
        self.model = model
        self.loss = nn.MarginRankingLoss(self.args.margin, reduction=self.args.reduction)

    def forward(self, pos_score, neg_score):
        pos_score = pos_score.view(-1)
        if self.args.reduction == 'sum':
            neg_score = neg_score.view(-1).view(len(pos_score), -1).mean(dim=1).view(-1)
        else:
            neg_score = neg_score.view(-1)
        label = torch.Tensor([1]).type_as(pos_score)
        loss = self.loss(pos_score, neg_score, label)
            
        return loss