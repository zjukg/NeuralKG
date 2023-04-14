import torch
import torch.nn as nn 
from .model import Model

class HAKE(Model):
    """`Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction`_ (HAKE), which maps entities into the polar coordinate system.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim * 2].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim * 3].
        phase_weight: Calculate phase score.
        modules_weight: Calculate modulus score.
    
    .. _Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction: https://arxiv.org/pdf/1911.09419.pdf
    """
    def __init__(self, args):
        super(HAKE, self).__init__(args)
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
            requires_grad=False,
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False,
        )
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        nn.init.uniform_(
            tensor = self.ent_emb.weight.data, 
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(),
        )
        
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 3)
        nn.init.uniform_(
            tensor = self.rel_emb.weight.data,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(),
        )

        nn.init.ones_(
            tensor=self.rel_emb.weight[:, self.args.emb_dim: 2*self.args.emb_dim],
        )
        nn.init.zeros_(
            tensor=self.rel_emb.weight[:, 2*self.args.emb_dim: 3*self.args.emb_dim]
        )

        self.phase_weight = nn.Parameter(
            torch.Tensor([self.args.phase_weight * self.embedding_range.item()])
        )
        self.modules_weight = nn.Parameter(
            torch.Tensor([self.args.modulus_weight])
        )

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - ||h_m \circ r_m- t_m||_2 - \lambda ||\sin((h_p + r_p - t_p)/2)||_1`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        phase_head, mod_head = torch.chunk(head_emb, 2, dim=-1)
        phase_tail, mod_tail = torch.chunk(tail_emb, 2, dim=-1)
        phase_rela, mod_rela, bias_rela = torch.chunk(relation_emb, 3, dim=-1)

        pi = 3.141592653589793
        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)
        phase_rela = phase_rela / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            phase_score = phase_head + (phase_rela - phase_tail)
        else:
            phase_score = (phase_head + phase_rela) - phase_tail
        
        mod_rela = torch.abs(mod_rela)
        bias_rela = torch.clamp(bias_rela, max=1)

        indicator = (bias_rela < -mod_rela)
        bias_rela[indicator] = -mod_rela[indicator]

        r_score = mod_head * (mod_rela + bias_rela) - mod_tail * (1 - bias_rela)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score /2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modules_weight

        return self.margin.item() - (phase_score + r_score)

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
        triples = batch['positive_sample']
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
        

