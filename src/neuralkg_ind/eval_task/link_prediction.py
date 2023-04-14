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
        ranks = ind_predict(batch, model)

    return ranks.float()

def ind_predict(batch, model):
    """Getting the ranking of positive samples in the other 50 negative samples.
        
    Args:
        batch: The batch of sample for test.
        model: The inductive model for training.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    head_triple = batch["head_sample"]
    head_scores = model(head_triple).squeeze(1).detach().cpu().numpy()
    head_target = batch["head_target"]
    head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target) + 1
    head_rank = torch.tensor(head_rank).squeeze(0)


    tail_triple = batch["tail_sample"]
    tail_scores = model(tail_triple).squeeze(1).detach().cpu().numpy()
    tail_target = batch["tail_target"]
    tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target) + 1
    tail_rank = torch.tensor(tail_rank).squeeze(0)

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

def link_predict_SEGNN(batch, kg, model, prediction="all"):
    """The evaluate task is predicting the head entity or tail entity in incomplete triples.
        
    Args:
        batch: The batch of the triples for validation or test.
        model: The KG model for training.
        predicion: mode of link prediction.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    ent_emb, rel_emb = model.aggragate_emb(kg)
    if prediction == "all":
        tail_ranks = tail_predict_SEGNN(batch, ent_emb, rel_emb, model)
        head_ranks = head_predict_SEGNN(batch, ent_emb, rel_emb, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict_SEGNN(batch, ent_emb, rel_emb, model)
    elif prediction == "tail":
        ranks = tail_predict_SEGNN(batch, ent_emb, rel_emb, model)

    return ranks.float()

def head_predict_SEGNN(batch, ent_emb, rel_emb, model):
    """Getting head entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The KG model for training.

    Returns:
        tensor: The rank of the head entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    head_idx = pos_triple[:, 0]
    tail_idx = pos_triple[:, 2]
    rel_idx = [pos_triple[:, 1][i] + 11 for i in range(len(pos_triple[:, 1]))]
    rel_idx = torch.tensor(rel_idx)
    filter_head = batch["filter_head"]
    pred_score = model.predictor.score_func(ent_emb[tail_idx], rel_emb[rel_idx], model.concat, ent_emb)
    return calc_ranks_SEGNN(head_idx, filter_head, pred_score)



def tail_predict_SEGNN(batch, ent_emb, rel_emb, model):
    """Getting tail entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The KG model for training.

    Returns:
        tensor: The rank of the tail entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    head_idx = pos_triple[:, 0]
    rel_idx = pos_triple[:, 1]
    tail_idx = pos_triple[:, 2]
    filter_tail = batch["filter_tail"]
    pred_score = model.predictor.score_func(ent_emb[head_idx], rel_emb[rel_idx], model.concat, ent_emb)
    return calc_ranks_SEGNN(tail_idx, filter_tail, pred_score)


def calc_ranks_SEGNN(idx, filter_label, pred_score):
    """Calculating triples score ranks.

    Args:
        idx ([type]): The id of the entity to be predicted.
        label ([type]): The id of existing triples, to calc filtered results.
        pred_score ([type]): The score of the triple predicted by the model.

    Returns:
        ranks: The rank of the triple to be predicted, dim [batch_size].
    """
    score = pred_score + filter_label
    size = filter_label.shape[0]
    pred_score1 = score[torch.arange(size), idx].unsqueeze(dim=1)
    compare_up = torch.gt(score, pred_score1)
    compare_low = torch.ge(score, pred_score1)

    ranking_up = compare_up.to(dtype=torch.float).sum(dim=1) + 1  # (bs, )
    ranking_low = compare_low.to(dtype=torch.float).sum(dim=1)  # include the pos one itself, no need to +1
    ranking = (ranking_up + ranking_low) / 2
    return ranking