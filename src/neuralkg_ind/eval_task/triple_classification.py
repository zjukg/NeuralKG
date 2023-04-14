import torch
import numpy as np

def classification(batch, model):
    """Calculating triple classification score.

    Args:
        batch: Positive sample and negative sample.
        model: The model for testing.

    Returns:
        score: The positive sample score and the negative sample score.
    """
    score = dict()

    pos_sample = batch["positive_sample"]
    score_pos = model(pos_sample)
    score_pos = score_pos.squeeze(1).detach().cpu().tolist()
    score["pos_scores"] = score_pos

    neg_sample = batch["negative_sample"]
    score_neg = model(neg_sample)
    score_neg = score_neg.squeeze(1).detach().cpu().tolist()
    score["neg_scores"] = score_neg

    return score