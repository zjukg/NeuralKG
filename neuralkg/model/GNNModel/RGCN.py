import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from neuralkg.model import DistMult


class RGCN(nn.Module):
    """`Modeling Relational Data with Graph Convolutional Networks`_ (RGCN), which use GCN framework to model relation data.

    Attributes:
        args: Model configuration parameters.
    
    .. _Modeling Relational Data with Graph Convolutional Networks: https://arxiv.org/pdf/1703.06103.pdf
    """

    def __init__(self, args):
        super(RGCN, self).__init__()
        self.args = args
        self.ent_emb = None
        self.rel_emb = None 
        self.RGCN = None 
        self.Loss_emb = None
        self.build_model()

    def build_model(self):
        """Initialize the RGCN model and embeddings 

        Args:
            ent_emb: Entity embedding, shape:[num_ent, emb_dim].
            rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
            RGCN: the relation graph convolution model.
        """
        self.ent_emb = nn.Embedding(self.args.num_ent,self.args.emb_dim)

        self.rel_emb = nn.Parameter(torch.Tensor(self.args.num_rel, self.args.emb_dim))

        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.RGCN = nn.ModuleList()
        for idx in range(self.args.num_layers):
            RGCN_idx = self.build_hidden_layer(idx)
            self.RGCN.append(RGCN_idx)
        
    def forward(self, graph, ent, rel, norm, triples, mode='single'):
        
        """The functions used in the training and testing phase

        Args:
            graph: The knowledge graph recorded in dgl.graph()
            ent: The entitiy ids sampled in triples
            rel: The relation ids sampled in triples
            norm: The edge norm in graph 
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        
        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = DistMult.score_func(self,head_emb, rela_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples    = batch['positive_sample']
        graph      = batch['graph']
        ent        = batch['entity']
        rel        = batch['rela']
        norm       = batch['norm']

        embedding = self.ent_emb(ent.squeeze())
        for layer in self.RGCN:
            embedding = layer(graph, embedding, rel, norm)
        self.Loss_emb = embedding
        head_emb, rela_emb, tail_emb = self.tri2emb(embedding, triples, mode)
        score = DistMult.score_func(self,head_emb, rela_emb, tail_emb, mode)

        return score

    def tri2emb(self, embedding, triples, mode="single"): #TODO:和XTransE合并
        
        """Get embedding of triples.
        
        This function get the embeddings of head, relation, and tail
        respectively. each embedding has three dimensions.

        Args:
            embedding(tensor): This embedding save the entity embeddings.            
            triples (tensor): This tensor save triples id, which dimension is 
                [triples number, 3].
            mode (str, optional): This arg indicates that the negative entity 
                will replace the head or tail entity. when it is 'single', it 
                means that entity will not be replaced. Defaults to 'single'.

        Returns:
            head_emb: Head entity embedding.
            rela_emb: Relation embedding.
            tail_emb: Tail entity embedding.
        """

        rela_emb = self.rel_emb[triples[:, 1]].unsqueeze(1)  # [bs, 1, dim]
        head_emb = embedding[triples[:, 0]].unsqueeze(1)  # [bs, 1, dim] 
        tail_emb = embedding[triples[:, 2]].unsqueeze(1)  # [bs, 1, dim]

        if mode == "head-batch" or mode == "head_predict":
            head_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        elif mode == "tail-batch" or mode == "tail_predict":
            tail_emb = embedding.unsqueeze(0)  # [1, num_ent, dim]

        return head_emb, rela_emb, tail_emb

    def build_hidden_layer(self, idx):
        """The functions used to initialize the RGCN model

        Args:
            idx: it`s used to identify rgcn layers. The last rgcn layer should use 
            relu as activation function.

        Returns:
            the relation graph convolution layer
        """
        act = F.relu if idx < self.args.num_layers - 1 else None
        return RelGraphConv(self.args.emb_dim, self.args.emb_dim, self.args.num_rel, "bdd",
                    num_bases=100, activation=act, self_loop=True,dropout=0.2 )
