import torch
import torch.nn.functional as F
import torch.nn as nn

class RGCN_Loss(nn.Module):

    def __init__(self, args, model):
        super(RGCN_Loss, self).__init__()
        self.args = args
        self.model = model
    
    def reg_loss(self): 
        return torch.mean(self.model.Loss_emb.pow(2)) + torch.mean(self.model.rel_emb.pow(2))

    def forward(self, score, labels):
         loss = F.binary_cross_entropy_with_logits(score, labels)
         regu = self.args.regularization * self.reg_loss()
         loss += regu
         return loss