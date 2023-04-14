import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class SimplE_Loss(nn.Module):
    def __init__(self, args, model):
        super(SimplE_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score):
        pos_score = -pos_score
        score = torch.cat((neg_score, pos_score), dim = -1) #shape:[bs, neg_num+1]
        loss = torch.sum(F.softplus(score)) + self.args.regularization * self.model.l2_loss()

        return loss


        
