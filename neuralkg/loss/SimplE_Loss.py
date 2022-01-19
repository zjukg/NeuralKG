import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class SimplE_Loss(nn.Module):
    def __init__(self, args, model):
        super(SimplE_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_sample, neg_sample, mode):
        neg_score = self.model(pos_sample, neg_sample, mode) #shape:[bs, neg_num]
        pos_score = -self.model(pos_sample).view(-1, 1) #shape:[bs]
        score = torch.cat((neg_score, pos_score), dim = -1) #shape:[bs, neg_num+1]
        loss = torch.sum(F.softplus(score)) + self.args.regularization * self.model.l2_loss()

        return loss


        
