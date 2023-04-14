import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
import os

class KBAT(nn.Module):

    """`Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs`_ (KBAT), 
        which introduces the attention to aggregate the neighbor node representation.

    Attributes:
        args: Model configuration parameters.
    
    .. _Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs: 
        https://arxiv.org/pdf/1906.01195.pdf
    """

    def __init__(self, args):
        super(KBAT,self).__init__()
        self.args = args
        self.entity_embeddings   = None
        self.relation_embeddings = None

        self.init_GAT_emb()
        self.init_ConvKB_emb()

    def init_GAT_emb(self):
        """Initialize the GAT model and embeddings 

        Args:
            ent_emb_out: Entity embedding, shape:[num_ent, emb_dim].
            rel_emb_out: Relation_embedding, shape:[num_rel, emb_dim].
            entity_embeddings: The final embedding used in ConvKB.
            relation_embeddings: The final embedding used in ConvKB.
            attentions, out_att: The graph attention layers.
        """
        self.num_ent = self.args.num_ent
        self.num_rel = self.args.num_rel
        self.emb_dim = self.args.emb_dim

        self.ent_emb_out = nn.Parameter(torch.randn(self.num_ent,self.emb_dim))
        self.rel_emb_out = nn.Parameter(torch.randn(self.num_rel,self.emb_dim))

        self.drop  = 0.3
        self.alpha = 0.2  

        self.nheads_GAT = 2
        self.out_dim = 100

        self.entity_embeddings = nn.Parameter(
            torch.randn(self.num_ent, self.out_dim * self.nheads_GAT))

        self.relation_embeddings = nn.Parameter(
            torch.randn(self.num_rel, self.out_dim * self.nheads_GAT))

        self.dropout_layer = nn.Dropout(self.drop)

        self.attentions = [GraphAttentionLayer(self.num_ent, 
                                                 self.emb_dim,
                                                 self.out_dim,
                                                 self.emb_dim,
                                                 dropout=self.drop,
                                                 alpha=self.alpha,
                                                 concat=True)
                           for _ in range(self.nheads_GAT)] 

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension 变换矩阵
        self.W = nn.Parameter(torch.zeros(
            size=(self.emb_dim, self.nheads_GAT * self.out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = GraphAttentionLayer(self.num_ent, 
                                             self.out_dim * self.nheads_GAT,
                                             self.out_dim * self.nheads_GAT,
                                             self.out_dim * self.nheads_GAT,
                                             dropout=self.drop,
                                             alpha=self.alpha,
                                             concat=False
                                             )

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.emb_dim, self.out_dim * self.nheads_GAT)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def init_ConvKB_emb(self): 
        """Initialize the ConvKB model.

        Args:
            conv_layer: The convolution layer.
            dropout: The dropout layer.
            ReLU: Relu activation function.
            fc_layer: The full connection layer.
        """
        self.conv_layer = nn.Conv2d(1, 50, (1,3))  
        self.dropout    = nn.Dropout(0.3)
        self.ReLU       = nn.ReLU()
        self.fc_layer   = nn.Linear(10000, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, triples, mode, adj_matrix=None, n_hop=None):
        """The functions used in the training and testing phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            mode: The mode indicates that the model will be used, when it 
            is 'GAT', it means graph attetion model, when it is 'ConvKB', 
            it means ConvKB model. 

        Returns:
            score: The score of triples.
        """
        if mode == 'GAT':  # gat
            score = self.forward_GAT(triples, adj_matrix, n_hop)

        else:
            score = self.forward_Con(triples, mode)
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
        score = self.forward_Con(triples, mode)
        return score

    def forward_Con(self, triples, mode):
        score = None

        if mode == 'ConvKB':
            head_emb = self.entity_embeddings[triples[:, 0]].unsqueeze(1)
            rela_emb = self.relation_embeddings[triples[:, 1]].unsqueeze(1) 
            tail_emb = self.entity_embeddings[triples[:, 2]].unsqueeze(1)
            score = self.cal_Con_score(head_emb, rela_emb, tail_emb)
                
        elif mode == 'head_predict':
            head_emb = self.entity_embeddings.unsqueeze(1)  # [1, num_ent, dim]
            for triple in triples:
                rela_emb = self.relation_embeddings[triple[1]].\
                                unsqueeze(0).tile(dims=(self.num_ent,1,1))
                tail_emb = self.entity_embeddings[triple[2]].\
                                unsqueeze(0).tile(dims=(self.num_ent,1,1))
                s = self.cal_Con_score(head_emb, rela_emb, tail_emb).t()
                
                if score == None:
                    score = s
                else:
                    score = torch.cat((score, s), dim=0)


        elif mode == 'tail_predict':
            tail_emb = self.entity_embeddings.unsqueeze(1)  # [1, num_ent, dim]
            for triple in triples:
                head_emb = self.entity_embeddings[triple[0]].\
                                unsqueeze(0).tile(dims=(self.num_ent,1,1))
                rela_emb = self.relation_embeddings[triple[1]].\
                                unsqueeze(0).tile(dims=(self.num_ent,1,1))
                s = self.cal_Con_score(head_emb, rela_emb, tail_emb).t()
                
                if score == None:
                    score = s
                else:
                    score = torch.cat((score, s), dim=0)

        return score

    def forward_GAT(self, triples, adj_matrix, n_hop):
        edge_list = adj_matrix[0] #边节点
        edge_type = adj_matrix[1] #边种类
        edge_list_nhop = torch.cat((n_hop[:, 3].unsqueeze(-1), 
                                        n_hop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat([n_hop[:, 1].unsqueeze(-1), 
                                        n_hop[:, 2].unsqueeze(-1)], dim=1)

        edge_emb = self.rel_emb_out[edge_type]
        self.ent_emb_out.data = F.normalize(self.ent_emb_out.data, p=2, dim=1).detach()
        edge_embed_nhop = self.rel_emb_out[edge_type_nhop[:, 0]] + \
                self.rel_emb_out[edge_type_nhop[:, 1]]
            
        ent_emb_out = torch.cat([att(self.ent_emb_out, edge_list, edge_emb, edge_list_nhop, 
                                edge_embed_nhop) for att in self.attentions], dim=1)
        ent_emb_out = self.dropout_layer(ent_emb_out)
        rel_emb_out = self.rel_emb_out.mm(self.W)
        edge_emb = rel_emb_out[edge_type]
        edge_embed_nhop = rel_emb_out[edge_type_nhop[:, 0]] + \
                rel_emb_out[edge_type_nhop[:, 1]]
        ent_emb_out = F.elu(self.out_att(ent_emb_out, edge_list, edge_emb,
                                edge_list_nhop, edge_embed_nhop))
            
        mask_indices = torch.unique(triples[:, 2])
        mask = torch.zeros(self.ent_emb_out.shape[0]).type_as(self.ent_emb_out)
        mask[mask_indices] = 1.0

        entities_upgraded = self.ent_emb_out.mm(self.W_entities)
        ent_emb_out = entities_upgraded + \
                mask.unsqueeze(-1).expand_as(ent_emb_out) * ent_emb_out
        ent_emb_out = F.normalize(ent_emb_out, p=2, dim=1)

        self.entity_embeddings.data = ent_emb_out.data
        self.relation_embeddings.data = rel_emb_out.data

        head_emb = ent_emb_out[triples[:, 0]]
        rela_emb = rel_emb_out[triples[:, 1]]
        tail_emb = ent_emb_out[triples[:, 2]]
        return self.cal_GAT_score(head_emb, rela_emb, tail_emb)

    def cal_Con_score(self, head_emb, rela_emb, tail_emb):

        """Calculating the score of triples with ConvKB model.

        Args:
            head_emb: The head entity embedding.
            rela_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """      

        conv_input = torch.cat((head_emb, rela_emb, tail_emb), dim=1)            
        batch_size= conv_input.shape[0]

        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        
        out_conv = self.conv_layer(conv_input)
        out_conv = self.ReLU(out_conv)
        out_conv = self.dropout(out_conv)
        out_conv = out_conv.squeeze(-1).view(batch_size, -1)
        score = self.fc_layer(out_conv)
        return score

    def cal_GAT_score(self, head_emb, relation_emb, tail_emb):

        """Calculating the score of triples with TransE model.

        Args:
            head_emb: The head entity embedding.
            rela_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """

        score = (head_emb + relation_emb) - tail_emb
        score = torch.norm(score, p=1, dim=1)
        return score

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """
    Special function for only sparse region backpropataion layer, similar to https://arxiv.org/abs/1710.10903
    """
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):

        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            grad_values = grad_output[edge_sources]

        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    """
    Special spmm final layer, similar to https://arxiv.org/abs/1710.10903.
    """
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class GraphAttentionLayer(nn.Module):
    
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903.
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   
        self.out_features = out_features 
        self.num_nodes = num_nodes       
        self.alpha = alpha        
        self.concat = concat
        self.nrela_dim = nrela_dim       

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'