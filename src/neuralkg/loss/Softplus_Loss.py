import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Softplus_Loss(nn.Module):
    """softplus loss. 
    Attributes:
        args: Some pre-set parameters, etc. 
        model: The KG model for training.
    """
    def __init__(self, args, model):
        super(Softplus_Loss, self).__init__()
        self.criterion = nn.Softplus()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, subsampling_weight=None):
        """Negative sampling loss Softplus_Loss. In math:
        
        \begin{aligned}
            L(\boldsymbol{Q}, \boldsymbol{W})=& \sum_{r(h, t) \in \Omega \cup \Omega^{-}} \log \left(1+\exp \left(-Y_{h r t} \phi(h, r, t)\right)\right) \\
            &+\lambda_1\|\boldsymbol{Q}\|_2^2+\lambda_2\|\boldsymbol{W}\|_2^2
        \end{aligned}
        
        Args:
            pos_score: The score of positive samples (with regularization if DualE).
            neg_score: The score of negative samples (with regularization if DualE).
        Returns:
            loss: The training loss for back propagation.
        """
        if self.args.model_name == 'DualE':
            p_score, pos_regul_1, pos_regul_2 = pos_score
            n_score, neg_regul_1, neg_regul_2 = neg_score
        score = torch.cat((-p_score,n_score))
        loss = torch.mean(self.criterion(score))

        if self.args.model_name == 'DualE':
            regularization1 = (pos_regul_1+neg_regul_1*self.args.num_neg)/(self.args.num_neg+1)*self.args.regularization
            regularization2 = (pos_regul_2+neg_regul_2*self.args.num_neg)/(self.args.num_neg+1)*self.args.regularization_two
            loss = loss+regularization1+regularization2

        return loss