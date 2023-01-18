import torch.nn as nn
import torch
import torch.nn.functional as F
from .model import Model
from IPython import embed


class TransH(Model):
    """`Knowledge Graph Embedding by Translating on Hyperplanes`_ (TransH), which apply the translation from head to tail entity in a
    relational-specific hyperplane in order to address its inability to model one-to-many, many-to-one, and many-to-many relations.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        norm_vector: Relation-specific projection matrix, shape:[num_rel, emb_dim]

    .. _Knowledge Graph Embedding by Translating on Hyperplanes: https://ojs.aaai.org/index.php/AAAI/article/view/8870
    """

    def __init__(self, args):
        super(TransH, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.norm_flag = args.norm_flag
        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False,
        )

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        self.norm_vector = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(
            tensor=self.ent_emb.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        nn.init.uniform_(
            tensor=self.rel_emb.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        nn.init.uniform_(
            tensor=self.norm_vector.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - \|e'_{h,r} + d_r - e'_{t,r}\|_{p}^2`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        if self.norm_flag:
            head_emb = F.normalize(head_emb, 2, -1)
            relation_emb = F.normalize(relation_emb, 2, -1)
            tail_emb = F.normalize(tail_emb, 2, -1)
        if mode == "head-batch" or mode == "head_predict":
            score = head_emb + (relation_emb - tail_emb)
        else:
            score = (head_emb + relation_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def forward(self, triples, negs=None, mode="single"):
        """The functions used in the training phase, same as TransE"""

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        norm_vector = self.norm_vector(triples[:, 1]).unsqueeze(
            dim=1
        )  # shape:[bs, 1, dim]
        head_emb = self._transfer(head_emb, norm_vector)
        tail_emb = self._transfer(tail_emb, norm_vector)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase, same as TransE"""

        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        norm_vector = self.norm_vector(triples[:, 1]).unsqueeze(
            dim=1
        )  # shape:[bs, 1, dim]
        head_emb = self._transfer(head_emb, norm_vector)
        tail_emb = self._transfer(tail_emb, norm_vector)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score

    def _transfer(self, emb, norm_vector):
        """Projecting entity embeddings onto the relation-specific hyperplane

        The formula for Projecting entity embeddings is :math:`e'_{r} = e - w_r^\Top e w_r`

        Args:
            emb: Entity embeddings, shape:[batch_size, emb_dim]
            norm_vector: Relation-specific projection matrix, shape:[num_rel, emb_dim]

        Returns:
            projected entity emb: Shape:[batch_size, emb_dim]

        """
        if self.norm_flag:
            norm_vector = F.normalize(norm_vector, p=2, dim=-1)
        return emb - torch.sum(emb * norm_vector, -1, True) * norm_vector
