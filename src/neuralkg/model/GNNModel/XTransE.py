import torch.nn as nn
import torch
from IPython import embed
from neuralkg.model.KGEModel.model import Model

class XTransE(Model):

    """`Explainable Knowledge Graph Embedding for Link Prediction with Lifestyles in e-Commerce`_ (XTransE), which introduces the attention to aggregate the neighbor node representation.

    Attributes:
        args: Model configuration parameters.
    
    .. _Explainable Knowledge Graph Embedding for Link Prediction with Lifestyles in e-Commerce: https://link.springer.com/content/pdf/10.1007%2F978-981-15-3412-6_8.pdf
    """

    def __init__(self, args):
        super(XTransE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):

        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        Args:
            margin: Caculate embedding_range and loss.
            embedding_range: Uniform distribution range.
            ent_emb: Entity embedding, shape:[num_ent, emb_dim].
            rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
        """

        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([6.0 / float(self.args.emb_dim).__pow__(0.5)]),
            requires_grad=False
        )
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, triples, neighbor=None, mask=None, negs=None, mode='single'):

        """Calculating the score of triples.
        
        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            neighbor: The neighbors of tail entities.
            mask: The mask of neighbor nodes
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.                                                                                                                                                                            

        Returns:
            score: The score of triples.
        """

        head = triples[:,0]
        rela = triples[:,1]
        tail = triples[:,2]

        if mode == 'tail-batch':
            tail = negs.squeeze(1)
        
        norm_emb_ent = nn.functional.normalize(self.ent_emb.weight, dim=1, p=2) # [ent, dim]
        norm_emb_rel = nn.functional.normalize(self.rel_emb.weight, dim=1, p=2) # [rel, dim]
        
        neighbor_tail_emb = norm_emb_ent[neighbor[:, :, 1]] # [batch, neighbor, dim]
        neighbor_rela_emb = norm_emb_rel[neighbor[:, :, 0]] # [batch, neighbor, dim]
        neighbor_head_emb = neighbor_tail_emb - neighbor_rela_emb 

        rela_emb = norm_emb_rel[rela] # [batch, dim]
        tail_emb = norm_emb_ent[tail] # [batch, dim]
        head_emb = norm_emb_ent[head]
        h_rt_embedding = tail_emb - rela_emb

        attention_rt = torch.zeros([self.args.train_bs, 200]).type_as(self.ent_emb.weight)
        attention_rt = (neighbor_head_emb * h_rt_embedding.unsqueeze(1)).sum(dim=2) * mask
        attention_rt = nn.functional.softmax(attention_rt, dim=1).unsqueeze(2)

        head_emb = head_emb + \
            torch.bmm(neighbor_head_emb.permute(0,2,1), attention_rt).reshape([-1,self.args.emb_dim])

        score = self.margin.item() - torch.norm(head_emb + rela_emb - tail_emb, p=2, dim=1)
        return score.unsqueeze(1)

    def transe_func(self, head_emb, rela_emb, tail_emb):
        
        """Calculating the score of triples with TransE model.

        Args:
            head_emb: The head entity embedding.
            rela_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """

        score = (head_emb + rela_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=2, dim=-1)
        return score

    def forward(self, triples, neighbor=None, mask=None, negs=None, mode='single'):

        """The functions used in the training and testing phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            neighbor: The neighbors of tail entities.
            mask: The mask of neighbor nodes
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        TransE_score = self.transe_func(head_emb, relation_emb, tail_emb)
        XTransE_score = self.score_func(triples, neighbor, mask, negs, mode)
        
        return TransE_score + XTransE_score

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
        score = self.transe_func(head_emb, relation_emb, tail_emb)
        
        return score



