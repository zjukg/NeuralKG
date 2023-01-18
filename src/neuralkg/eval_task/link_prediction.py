import torch
import os
from IPython import embed
import numpy as np

def link_predict(batch, model, prediction="all", model_name=None):
    """The evaluate task is predicting the head entity or tail entity in incomplete triples.
        
    Args:
        batch: The batch of the triples for validation or test.
        model: The KG model for training.
        predicion: mode of link prediction.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    if prediction == "all":
        tail_ranks = tail_predict(batch, model)
        head_ranks = head_predict(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict(batch, model)
    elif prediction == "tail":
        ranks = tail_predict(batch, model)
    elif prediction == "ind":
        ranks = ind_predict(batch, model, model_name)

    return ranks.float()

def ind_predict(batch, model, model_name=None):
    head_triple = batch["head_sample"]
    head_scores = model(head_triple).squeeze(1).detach().cpu().numpy()
    head_target = batch["head_target"]
    if head_target != 10000:  
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target) + 1
        head_rank = torch.tensor(head_rank).squeeze(0)
    else:
        head_rank = torch.tensor([10000])

    tail_triple = batch["tail_sample"]
    tail_scores = model(tail_triple).squeeze(1).detach().cpu().numpy()
    tail_target = batch["tail_target"]
    if tail_target != 10000:
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target) + 1
        tail_rank = torch.tensor(tail_rank).squeeze(0)
    else:
        tail_rank = torch.tensor([10000])
    return torch.cat([head_rank, tail_rank])


def head_predict(batch, model):
    """Getting head entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The KG model for training.

    Returns:
        tensor: The rank of the head entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0]
    label = batch["head_label"]
    pred_score = model.get_score(batch, "head_predict")
    return calc_ranks(idx, label, pred_score)


def tail_predict(batch, model):
    """Getting tail entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The KG model for training.

    Returns:
        tensor: The rank of the tail entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 2]
    label = batch["tail_label"]
    pred_score = model.get_score(batch, "tail_predict")
    return calc_ranks(idx, label, pred_score)


def calc_ranks(idx, label, pred_score):
    """Calculating triples score ranks.

    Args:
        idx ([type]): The id of the entity to be predicted.
        label ([type]): The id of existing triples, to calc filtered results.
        pred_score ([type]): The score of the triple predicted by the model.

    Returns:
        ranks: The rank of the triple to be predicted, dim [batch_size].
    """

    b_range = torch.arange(pred_score.size()[0])
    target_pred = pred_score[b_range, idx]
    pred_score = torch.where(label.bool(), -torch.ones_like(pred_score) * 10000000, pred_score)
    pred_score[b_range, idx] = target_pred

    ranks = (
        1
        + torch.argsort(
            torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
        )[b_range, idx]
    )
    return ranks