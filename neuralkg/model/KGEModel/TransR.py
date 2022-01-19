import torch.nn as nn
import torch
import torch.nn.functional as F
from .model import Model
from IPython import embed


class TransR(Model):
    """Learning Entity and Relation Embeddings for Knowledge Graph Completion`_ (TransR), which building entity and relation embeddings in separate entity space and relation spaces

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        transfer_matrix: Transfer entity and relation embedding, shape:[num_rel, emb_dim*emb_dim]

    .. _Translating Embeddings for Modeling Multi-relational Data: http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/
    """

    def __init__(self, args):
        super(TransR, self).__init__(args)
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
        self.transfer_matrix = nn.Embedding(
            self.args.num_rel, self.args.emb_dim * self.args.emb_dim
        )
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
        diag_matrix = torch.eye(self.args.emb_dim)
        diag_matrix = diag_matrix.flatten().repeat(self.args.num_rel, 1)
        self.transfer_matrix.weight.data = diag_matrix

        # nn.init.uniform_(tensor=self.transfer_matrix.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - \| M_{r} {e}_h + r_r - M_{r}e_t \|_{p}^2`

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
        """The functions used in the training phase, calculate triple score."""
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        rel_transfer = self.transfer_matrix(triples[:, 1])  # shape:[bs, dim]
        head_emb = self._transfer(head_emb, rel_transfer, mode)
        tail_emb = self._transfer(tail_emb, rel_transfer, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase, predict triple score."""
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        rel_transfer = self.transfer_matrix(triples[:, 1])  # shape:[bs, dim]
        head_emb = self._transfer(head_emb, rel_transfer, mode)
        tail_emb = self._transfer(tail_emb, rel_transfer, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def _transfer(self, emb, rel_transfer, mode):
        """Transfer entity embedding with relation-specific matrix.

        Args:
            emb: Entity embeddings, shape:[batch_size, emb_dim]
            rel_transfer: Relation-specific projection matrix, shape:[batch_size, emb_dim]
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            transfered entity emb: Shape:[batch_size, emb_dim]

        """
        rel_transfer = rel_transfer.view(-1, self.args.emb_dim, self.args.emb_dim)
        rel_transfer = rel_transfer.unsqueeze(dim=1)
        emb = emb.unsqueeze(dim=-2)
        emb = torch.matmul(emb, rel_transfer)
        return emb.squeeze(dim=-2)
