import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Adv_Loss(nn.Module):
    def __init__(self, args, model):
        super(Adv_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score, subsampling_weight=None):
        if self.args.negative_adversarial_sampling:
            neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                        * F.logsigmoid(-neg_score)).sum(dim=1)  #shape:[bs]
        else:
            neg_score = F.logsigmoid(-neg_score).mean(dim = 1)

        pos_score = F.logsigmoid(pos_score).view(neg_score.shape[0]) #shape:[bs]
        # from IPython import embed;embed();exit()

        if self.args.use_weight:
            positive_sample_loss = - (subsampling_weight * pos_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * neg_score).sum()/subsampling_weight.sum()
        else:
            positive_sample_loss = - pos_score.mean()
            negative_sample_loss = - neg_score.mean()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult' or self.args.model_name == 'BoxE':
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            # embed();exit()
            loss = loss + regularization
        return loss
    
    def normalize(self):
        regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
        return regularization