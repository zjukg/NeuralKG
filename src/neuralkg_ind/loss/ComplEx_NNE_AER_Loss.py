import torch
import torch.nn as nn


class ComplEx_NNE_AER_Loss(nn.Module):
    def __init__(self, args, model):
        super(ComplEx_NNE_AER_Loss, self).__init__()
        self.args = args
        self.model = model
        self.rule_p, self.rule_q = model.rule
        self.confidence = model.conf
        
    def forward(self, pos_score, neg_score):
        logistic_neg = torch.log(1 + torch.exp(neg_score)).sum(dim=1)
        logistic_pos = torch.log(1 + torch.exp(-pos_score)).sum(dim=1)
        logistic_loss = logistic_neg + logistic_pos

        re_p, im_p = self.model.rel_emb(self.rule_p).chunk(2, dim=-1)
        re_q, im_q = self.model.rel_emb(self.rule_q).chunk(2, dim=-1)
        entail_loss_re = self.args.mu * torch.sum(
            self.confidence * (re_p - re_q).clamp(min=0).sum(dim=-1)
        )
        entail_loss_im = self.args.mu * torch.sum(
            self.confidence * (im_p - im_q).pow(2).sum(dim=-1)
        )
        entail_loss = entail_loss_re + entail_loss_im
        loss = logistic_loss + entail_loss
        # return loss
        if self.args.regularization != 0.0:
            # Use L2 regularization for ComplEx_NNE_AER
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p=2) ** 2
                + self.model.rel_emb.weight.norm(p=2) ** 2
            )
            loss = loss + regularization
        loss = loss.mean()
        return loss
