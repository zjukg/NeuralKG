import torch
import os
from IPython import embed

def RGCN_predict(batch, model, predicion='all'):
    if predicion == 'all':
        tail_ranks = head_predict(batch, model)
        head_ranks = tail_predict(batch, model)
        ranks = torch.cat([tail_ranks,head_ranks])
    elif predicion == 'tail':
        ranks = head_predict(batch, model)
    elif predicion == 'head':
        ranks = tail_predict(batch, model)

    return ranks.float()

def head_predict(batch,model):
    pos_triple = batch['positive_sample']
    idx        = pos_triple[:, 0]
    label      = batch['head_label']
    graph      = batch['graph']
    ent        = batch['entity']
    rel        = batch['rela']
    norm       = batch['norm'] 
    return calc_ranks(graph, ent, rel, norm, pos_triple, idx, label, "head-batch", model)

def tail_predict(batch,model):
    pos_triple = batch['positive_sample']
    idx        = pos_triple[:, 2]
    label      = batch['tail_label']
    graph      = batch['graph']
    ent        = batch['entity']
    rel        = batch['rela']
    norm       = batch['norm'] 

    return calc_ranks(graph, ent, rel, norm, pos_triple, idx, label, "tail-batch", model)

def calc_ranks(graph, ent, rel, norm, pos_triple, idx, label, mode, model):
    pred = model(graph, ent, rel, norm, pos_triple, mode=mode)  #TODO： 这里怎么改合适？ 是重新写一个还是？
    b_range = torch.arange(pred.size()[0])
    target_pred = pred[b_range, idx]
    pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
    pred[b_range, idx] = target_pred

    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                    dim=1, descending=False)[b_range, idx]
    return ranks