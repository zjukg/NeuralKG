import torch.nn as nn
import torch
from torch.autograd import Variable
from .model import Model

class BoxE(Model):
    
    """`A Box Embedding Model for Knowledge Base Completion`_ (BoxE), which represents the bump embedding as translations in the super rectangle space.

    Attributes:
        args: Model configuration parameters.
    
    .. _A Box Embedding Model for Knowledge Base Completion: https://arxiv.org/pdf/2007.06267.pdf
    """

    def __init__(self, args):
        super(BoxE, self).__init__(args)
        self.args = args
        self.arity = None
        self.order = None
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb(args)

    def init_emb(self,args):

        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        Args:
            arity: The maximum ary of the knowledge graph.
            epsilon: Caculate embedding_range.
            order: The distance order of score.
            margin: Caculate embedding_range and loss.
            embedding_range: Uniform distribution range.
            ent_emb: Entity embedding, shape:[num_ent, emb_dim * 2].
            rel_emb: Relation_embedding, shape:[num_rel, emb_dim * arity * 2].
        """
        
        self.arity = 2
        self.epsilon = 2.0
        self.order = self.args.dis_order
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False
        )
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim*2)
        nn.init.uniform_(tensor=self.ent_emb.weight.data[:, :self.args.emb_dim], a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.ent_emb.weight.data[:, self.args.emb_dim:], a=-self.embedding_range.item(), b=self.embedding_range.item())
        size_factor = self.arity * 2
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * size_factor)

    def forward(self, triples, negs=None, mode='single'):

        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """

        head_emb_raw, relation_emb, tail_emb_raw = self.tri2emb(triples, negs, mode)
        head_emb = head_emb_raw[:, :, :self.args.emb_dim] + tail_emb_raw[:, :, self.args.emb_dim:] 
        tail_emb = tail_emb_raw[:, :, :self.args.emb_dim] + head_emb_raw[:, :, self.args.emb_dim:]
        score = self.score_func(head_emb, relation_emb, tail_emb)
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
        head_emb_raw, relation_emb, tail_emb_raw = self.tri2emb(triples, mode=mode)
        head_emb = head_emb_raw[:, :, :self.args.emb_dim] + tail_emb_raw[:, :, self.args.emb_dim:] 
        tail_emb = tail_emb_raw[:, :, :self.args.emb_dim] + head_emb_raw[:, :, self.args.emb_dim:]
        score = self.score_func(head_emb, relation_emb, tail_emb)
        return score
    
    def score_func(self, head_emb, relation_emb, tail_emb):

        """Calculate the score of the triple embedding.

        Args:
            head_emb: The embedding of head entity.
            relation_emb:The embedding of relation.
            tail_emb: The embedding of tail entity.

        Returns:
            score: The score of triples.
        """

        box_bas, box_del = torch.chunk(relation_emb, 2, dim = -1)
        box_sec = box_bas + 0.5 * box_del
        box_fir = box_bas - 0.5 * box_del
        box_low = torch.min(box_fir, box_sec)
        box_hig = torch.max(box_fir, box_sec)
        head_low, tail_low = torch.chunk(box_low, 2, dim = -1)
        head_hig, tail_hig = torch.chunk(box_hig, 2, dim = -1)
        head_score = self.calc_score(head_emb, head_low, head_hig, self.order)
        tail_score = self.calc_score(tail_emb, tail_low, tail_hig, self.order)
        score = self.margin.item() - (head_score + tail_score)
        return score
    
    def calc_score(self, ent_emb, box_low, box_hig, order = 2):
        
        """Calculate the norm of distance.

        Args:
            ent_emb: The embedding of entity.
            box_low: The lower boundaries of the super rectangle.
            box_hig: The upper boundaries of the super rectangle.
            order: The order of this distance.

        Returns:
            The norm of distance.
        """

        return torch.norm(self.dist(ent_emb, box_low, box_hig), p=order, dim=-1)
    
    def dist(self, ent_emb, lb, ub):

        """Calculate the distance.

        This function calculate the distance between the entity
        and the super rectangle. If the entity is in its target
        box, distance inversely correlates with box size, to
        maintain low distance inside large boxes and provide a
        gradient to keep points inside; if the entity is not in
        its target box, box size linearly correlates with distance,
        to penalize points outside larger boxes more severely.

        Args:
            ent_emb: The embedding of entity.
            lb: The lower boundaries of the super rectangle.
            ub: The upper boundaries of the super rectangle.

        Returns:
            The distance between entity and super rectangle.
        """

        c = (lb + ub) / 2
        w = ub - lb + 1
        k = 0.5 * (w - 1) * (w - 1 / w)
        return torch.where(torch.logical_and(torch.ge(ent_emb, lb), torch.le(ent_emb, ub)),
                        torch.abs(ent_emb - c) / w,
                        torch.abs(ent_emb - c) * w - k)
    



