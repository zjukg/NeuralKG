import torch
import torch.nn.functional as F
import torch.nn as nn

class KBAT_Loss(nn.Module):

    def __init__(self, args, model):
        super(KBAT_Loss, self).__init__()
        self.args = args
        self.model = model
        self.GAT_loss = nn.MarginRankingLoss(self.args.margin)
        self.Con_loss = nn.SoftMarginLoss()
      
    def forward(self, model, score, neg_score=None, label=None):
        if model == 'GAT':
            y     = -torch.ones( 2 * self.args.num_neg * self.args.train_bs).type_as(score)
            score = torch.tile(score, (2*self.args.num_neg, 1)).reshape(-1)
            loss  = self.GAT_loss(score, neg_score, y)
        elif model == 'ConvKB':
            loss = self.Con_loss(score.view(-1), label.view(-1))

        return loss