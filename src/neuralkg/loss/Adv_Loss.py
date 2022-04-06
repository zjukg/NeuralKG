import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Adv_Loss(nn.Module):
    """Negative sampling loss with self-adversarial training. 

    Attributes:
        args: Some pre-set parameters, such as self-adversarial temperature, etc. 
        model: The KG model for training.
    """
    def __init__(self, args, model):
        super(Adv_Loss, self).__init__()
        self.args = args
        self.model = model
    def forward(self, pos_score, neg_score, subsampling_weight=None):
        """Negative sampling loss with self-adversarial training. In math:
        
        L=-\log \sigma\left(\gamma-d_{r}(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^{n} p\left(h_{i}^{\prime}, r, t_{i}^{\prime}\right) \log \sigma\left(d_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)-\gamma\right)
        
        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            subsampling_weight: The weight for correcting pos_score and neg_score.

        Returns:
            loss: The training loss for back propagation.
        """

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

        if self.args.model_name == 'ComplEx' or self.args.model_name == 'DistMult' or self.args.model_name == 'BoxE' or self.args.model_name=="IterE":
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
            # embed();exit()
            loss = loss + regularization
        return loss
    
    def normalize(self):
        """calculating the regularization.
        """
        regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p = 3)**3 + \
                self.model.rel_emb.weight.norm(p = 3)**3
            )
        return regularization