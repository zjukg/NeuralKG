import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Logsig_Loss(nn.Module):  ##NOTE: 注意 Margin 改为了 Logsig 
    def __init__(self, args, model):
        super(Logsig_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score):
        neg_score = F.logsigmoid(-neg_score)  #shape:[bs]
        pos_score = F.logsigmoid(pos_score) #shape:[bs, 1]
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.args.model_name == "XTransE":
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 1) + \
                self.model.rel_emb.weight.norm(p = 1)
            )

        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            loss = loss + regularization
            
        return loss