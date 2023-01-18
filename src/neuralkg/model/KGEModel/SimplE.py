import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .model import Model
from IPython import embed


class SimplE(Model):
    """`SimplE Embedding for Link Prediction in Knowledge Graphs`_ (SimpleE), which presents a simple enhancement of CP (which we call SimplE) to allow the two embeddings of each entity to be learned dependently.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_h_emb: Entity embedding, shape:[num_ent, emb_dim].
        ent_t_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
        rel_inv_emb: Inverse Relation_embedding, shape:[num_rel, emb_dim].

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs: http://papers.neurips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self, args):
        super(SimplE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution."""
        self.ent_h_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.ent_t_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        self.rel_inv_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        sqrt_size = 6.0 / math.sqrt(self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_h_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.ent_t_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-sqrt_size, b=sqrt_size)
        nn.init.uniform_(tensor=self.rel_inv_emb.weight.data, a=-sqrt_size, b=sqrt_size)

    def score_func(self, hh_emb, rel_emb, tt_emb, ht_emb, rel_inv_emb, th_emb):
        """Calculating the score of triples.

        Args:
            hh_emb: The head entity embedding on embedding h.
            rel_emb: The relation embedding.
            tt_emb: The tail entity embedding on embedding t.
            ht_emb: The tail entity embedding on embedding h.
            rel_inv_emb: The tail entity embedding.
            th_emb: The head entity embedding on embedding t.

        Returns:
            score: The score of triples.
        """
        # return -(torch.sum(head_emb * relation_emb * tail_emb, -1) + \
        #     torch.sum(head_emb * rel_inv_emb * tail_emb, -1))/2
        scores1 = torch.sum(hh_emb * rel_emb * tt_emb, dim=-1)
        scores2 = torch.sum(ht_emb * rel_inv_emb * th_emb, dim=-1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def l2_loss(self):
        return (self.ent_h_emb.weight.norm(p = 2) ** 2 + \
            self.ent_t_emb.weight.norm(p = 2) ** 2 + \
            self.rel_emb.weight.norm(p = 2) ** 2 + \
            self.rel_inv_emb.weight.norm(p = 2) ** 2)
    
    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        rel_emb, rel_inv_emb, hh_emb, th_emb, ht_emb, tt_emb = self.get_emb(triples, negs, mode)
        return self.score_func(hh_emb, rel_emb, tt_emb, ht_emb, rel_inv_emb, th_emb)
    
    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        rel_emb, rel_inv_emb, hh_emb, th_emb, ht_emb, tt_emb = self.get_emb(triples, mode=mode)
        return self.score_func(hh_emb, rel_emb, tt_emb, ht_emb, rel_inv_emb, th_emb)
    
    def get_emb(self, triples, negs=None, mode='single'):
        if mode == 'single':
            rel_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            rel_inv_emb = self.rel_inv_emb(triples[:, 1]).unsqueeze(1)
            hh_emb = self.ent_h_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            th_emb = self.ent_t_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            ht_emb = self.ent_h_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
            tt_emb = self.ent_t_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
        elif mode == 'head-batch' or mode == "head_predict":
            if negs is None:  # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
                hh_emb = self.ent_h_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
                th_emb = self.ent_t_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                hh_emb = self.ent_h_emb(negs)  # [bs, num_neg, dim]
                th_emb = self.ent_t_emb(negs)  # [bs, num_neg, dim]

            rel_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            rel_inv_emb = self.rel_inv_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            ht_emb = self.ent_h_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
            tt_emb = self.ent_t_emb(triples[:, 2]).unsqueeze(1)  # [bs, 1, dim]
        elif mode == 'tail-batch' or mode == "tail_predict":
            if negs is None:  # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
                ht_emb = self.ent_h_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
                tt_emb = self.ent_t_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                ht_emb = self.ent_h_emb(negs)  # [bs, num_neg, dim]
                tt_emb = self.ent_t_emb(negs)  # [bs, num_neg, dim]

            rel_emb = self.rel_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            rel_inv_emb = self.rel_inv_emb(triples[:, 1]).unsqueeze(1)  # [bs, 1, dim]
            hh_emb = self.ent_h_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]
            th_emb = self.ent_t_emb(triples[:, 0]).unsqueeze(1)  # [bs, 1, dim]

        return rel_emb, rel_inv_emb, hh_emb, th_emb, ht_emb, tt_emb

