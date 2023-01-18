import torch.nn as nn
import torch
from .model import Model
from IPython import embed
import pdb


class RugE(Model):
    """`Knowledge Graph Embedding with Iterative Guidance from Soft Rules`_ (RugE), which is a novel paradigm of KG embedding with iterative guidance from soft rules.

    Attributes:
        args: Model configuration parameters.
        epsilon: Caculate embedding_range.
        margin: Caculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
    
    .. _Knowledge Graph Embedding with Iterative Guidance from Soft Rules: https://ojs.aaai.org/index.php/AAAI/article/view/11918
    """
    def __init__(self, args):
        super(RugE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """       
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`Re(< wr, es, e¯o >)`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        return torch.sum(
            re_head * re_tail * re_relation
            + im_head * im_tail * re_relation
            + re_head * im_tail * im_relation
            - im_head * re_tail * im_relation,
            -1
        )
        

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score