import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed

class Cross_Entropy_Loss(nn.Module):
    """Binary CrossEntropyLoss 

    Attributes:
        args: Some pre-set parameters, etc 
        model: The KG model for training.
    """
    def __init__(self, args, model):
        super(Cross_Entropy_Loss, self).__init__()
        self.args = args
        self.model = model
        self.loss = torch.nn.BCELoss()

    def forward(self, pred, label):
        """Creates a criterion that measures the Binary Cross Entropy between the target and
        the input probabilities. In math:

        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],
        
        Args:
            pred: The score of all samples.
            label: Vectors used to distinguish positive and negative samples.
        Returns:
            loss: The training loss for back propagation.
        """
        loss = self.loss(pred, label)
        return loss