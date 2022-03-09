import torch
import torch.nn as nn
from .model import Model
from IPython import embed
from torch.autograd import Variable


class ConvE(Model):

    """`Convolutional 2D Knowledge Graph Embeddings`_ (ConvE), which use a 2D convolution network for embedding representation.

    Attributes:
        args: Model configuration parameters.
    
    .. _Convolutional 2D Knowledge Graph Embeddings: https://arxiv.org/pdf/1707.01476.pdf
    """

    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.args = args
        self.emb_ent = None
        self.emb_rel = None

        self.init_emb(args)

    def init_emb(self,args):

        """Initialize the convolution layer and embeddings .

        Args:
            conv1: The convolution layer.
            fc: The full connection layer.
            bn0, bn1, bn2: The batch Normalization layer.
            inp_drop, hid_drop, feg_drop: The dropout layer.
            emb_ent: Entity embedding, shape:[num_ent, emb_dim].
            emb_rel: Relation_embedding, shape:[num_rel, emb_dim].
        """

        self.emb_dim1 = self.args.emb_shape
        self.emb_dim2 = self.args.emb_dim // self.emb_dim1
        self.emb_ent = torch.nn.Embedding(self.args.num_ent, self.args.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(self.args.num_rel, self.args.emb_dim, padding_idx=0)
        torch.nn.init.xavier_normal_(self.emb_ent.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)
        # Setting dropout
        self.inp_drop = torch.nn.Dropout(self.args.inp_drop)
        self.hid_drop = torch.nn.Dropout(self.args.hid_drop)
        self.feg_drop = torch.nn.Dropout2d(self.args.fet_drop)
        # Setting net model
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.args.emb_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.args.num_ent)))
        self.fc = torch.nn.Linear(self.args.hid_size,self.args.emb_dim)

    def score_func(self, head_emb, relation_emb, choose_emb = None):

        """Calculate the score of the triple embedding.

        This function calculate the score of the embedding.
        First, the entity and relation embeddings are reshaped
        and concatenated; the resulting matrix is then used as
        input to a convolutional layer; the resulting feature
        map tensor is vectorised and projected into a k-dimensional
        space.

        Args:
            head_emb: The embedding of head entity.
            relation_emb:The embedding of relation.

        Returns:
            score: Final score of the embedding.
        """

        stacked_inputs = torch.cat([head_emb, relation_emb], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feg_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, self.emb_ent.weight.transpose(1,0)) if choose_emb == None \
            else torch.mm(x, choose_emb.transpose(1, 0)) 
        x += self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, triples):

        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].

        Returns:
            score: The score of triples.
        """
        
        head_emb = self.emb_ent(triples[:, 0]).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rela_emb = self.emb_rel(triples[:, 1]).view(-1, 1, self.emb_dim1, self.emb_dim2)
        score = self.score_func(head_emb, rela_emb)
        return score

    def get_score(self, batch, mode="tail_predict"):

        """The functions used in the testing phase

        Args:
            batch: A batch of data.

        Returns:
            score: The score of triples.
        """
        
        triples = batch["positive_sample"]
        head_emb = self.emb_ent(triples[:, 0]).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rela_emb = self.emb_rel(triples[:, 1]).view(-1, 1, self.emb_dim1, self.emb_dim2)
        score = self.score_func(head_emb, rela_emb)
        return score


