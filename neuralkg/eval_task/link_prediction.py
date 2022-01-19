import torch
import os
from IPython import embed


def link_predict(batch, model, predicion="all"):
    if predicion == "all":
        tail_ranks = head_predict(batch, model)
        head_ranks = tail_predict(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif predicion == "head":
        ranks = head_predict(batch, model)
    elif predicion == "tail":
        ranks = tail_predict(batch, model)

    return ranks.float()


def head_predict(batch, model):
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0]
    label = batch["head_label"]
    pred_score = model.get_score(batch, "head_predict")
    return calc_ranks(idx, label, pred_score)


def tail_predict(batch, model):
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 2]
    label = batch["tail_label"]
    pred_score = model.get_score(batch, "tail_predict")
    return calc_ranks(idx, label, pred_score)


def calc_ranks(idx, label, pred_score):
    """Calculating triples score ranks

    Args:
        idx ([type]): The id of the entity to be predicted
        label ([type]): The id of existing triples, to calc filtered results
        pred_score ([type]): The score of the triple predicted by the model

    Returns:
        tensor: The rank of the triple to be predicted, dim [batch_size]
    """

    b_range = torch.arange(pred_score.size()[0])
    target_pred = pred_score[b_range, idx]
    pred_score = torch.where(label.byte(), -torch.ones_like(pred_score) * 10000000, pred_score)
    pred_score[b_range, idx] = target_pred

    ranks = (
        1
        + torch.argsort(
            torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
        )[b_range, idx]
    )
    return ranks