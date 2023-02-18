import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model
from .RGCN import RelGraphConv
from neuralkg.utils.tools import *

class Grail(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ent_emb = None
        self.rel_emb = None 

        self.gnn = RGCN(args = args, basiclayer = RelAttGraphConv)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.rel_emb_dim, sparse=False)

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.args.num_layers * self.args.emb_dim + self.args.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.args.num_layers * self.args.emb_dim + self.args.rel_emb_dim, 1)

    def forward(self, data):
        g, rel_labels = data
        g = dgl.batch(g)
        g.ndata['h'], _ = self.gnn(g)

        g_out = dgl.mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.num_layers * self.args.emb_dim),
                               head_embs.view(-1, self.args.num_layers * self.args.emb_dim),
                               tail_embs.view(-1, self.args.num_layers * self.args.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.num_layers * self.args.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output

class RGCN(Model):
    def __init__(self, args, basiclayer):
        super(RGCN, self).__init__(args)

        self.args = args
        self.basiclayer = basiclayer
        self.inp_dim = args.inp_dim
        self.emb_dim = args.emb_dim
        self.has_attn = args.has_attn
        
        self.attn_rel_emb = None
        self.attn_rel_emb_dim = args.attn_rel_emb_dim
        
        self.init_emb()

        self.build_model()

    def init_emb(self):

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.args.num_rel, self.attn_rel_emb_dim, sparse=False)

        aggregator_type = self.args.gnn_agg_type.upper()+"Aggregator"
        aggregator_class = import_class(f"neuralkg.model.{aggregator_type}")
        self.aggregator = aggregator_class(self.emb_dim)

        # create initial features
        self.features = torch.arange(self.inp_dim)

    def build_hidden_layer(self, idx): 
        input_flag = True if idx == 0 else False
        input_emb = self.inp_dim if idx == 0 else self.emb_dim
        return self.basiclayer(self.args, input_emb, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.args.aug_num_rels,
                        self.args.num_bases, None, F.relu, self.args.dropout, self.args.edge_dropout,
                        is_input_layer=input_flag, has_attn=self.has_attn)

    def forward(self, graph, rela=None):
        for layer in self.layers:
            rela = layer(graph, rela, self.attn_rel_emb)
        return graph.ndata.pop('h'), rela

class RelAttGraphConv(RelGraphConv):
    def __init__(self, args, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False):
        super().__init__(args, inp_dim, out_dim, 0, None, 0, bias=False, activation=activation, 
                        self_loop=False, dropout=dropout, layer_norm=False,)
        self.bias = bias
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases

        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.aggregator = aggregator

        if self.bias: 
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        if self.num_bases <= 0 or self.num_bases > self.num_rels: 
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))
     
        self.edge_dropout = nn.Dropout(edge_dropout)

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None):
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1)).type_as(weight)
        input_ = 'feat' if self.is_input_layer else 'h'

        def message(edges):
            w = weight.index_select(0, edges.data['type'])
            msg = edges.data['w'] * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)
            if self.has_attn:
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']), attn_rel_emb(edges.data['label'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1))
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(message, self.aggregator, None)

    def forward(self, g, rel_emb=None, attn_rel_emb=None):
        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)
        
        return rel_emb
