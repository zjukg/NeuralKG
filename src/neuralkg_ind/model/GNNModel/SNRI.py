import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Grail import RGCN
from .layer import BatchGRU
from .RGCN import RelGraphConv
from neuralkg_ind.model import TransE, DistMult

class SNRI(nn.Module):
    """`Subgraph Neighboring Relations Infomax for Inductive Link Prediction on Knowledge Graphs`_ (SNRI), which sufficiently 
        exploits complete neighboring relationsfrom two aspects and apply mutual information (MI) maximization for knowledge graph.

    Attributes:
        args: Model configuration parameters.
        gnn: RGCN model.
        rel_emb: Relation embedding, shape: [num_rel + 1, inp_dim].
        ent_padding: Entity padding, shape: [1, sem_dim].
        w_rel2ent: Weight matrix of relation to entity.

    .. _Subgraph Neighboring Relations Infomax for Inductive Link Prediction on Knowledge Graphs: https://arxiv.org/abs/2208.00850
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.gnn = RGCN(args = args, basiclayer = RelCompGraphConv)
        self.rel_emb = nn.Embedding(self.args.num_rel + 1, self.args.inp_dim, sparse=False, padding_idx=self.args.num_rel)
        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.args.sem_dim).uniform_(-1, 1))
        if self.args.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.args.inp_dim, self.args.sem_dim)
        elif self.args.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.args.inp_dim, self.args.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.args.nei_rels_dropout)
        self.dropout = nn.Dropout(self.args.dropout)
        self.softmax = nn.Softmax(dim=1)

        if self.args.add_ht_emb:    
            self.fc_layer = nn.Linear(3 * self.args.num_layers * self.args.emb_dim + self.args.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.args.num_layers * self.args.emb_dim + self.args.rel_emb_dim, 1)

        if self.args.decoder_model:
            self.fc_layer = nn.Linear(2 * self.args.num_layers * self.args.emb_dim, 1)
        
        if self.args.nei_rel_path:
            self.fc_layer = nn.Linear(3 * self.args.num_layers * self.args.emb_dim + 2 * self.args.emb_dim, 1)
            self.disc = Discriminator(self.args.num_layers * self.args.emb_dim + self.args.emb_dim, self.args.num_layers * self.args.emb_dim + self.args.emb_dim)
        else:
            self.disc = Discriminator(self.args.num_layers * self.args.emb_dim , self.args.num_layers * self.args.emb_dim)

        if self.args.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)

        self.rnn = torch.nn.GRU(self.args.emb_dim, self.args.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.args.num_layers * self.args.emb_dim )

        self.W_o = nn.Linear(self.args.num_layers * self.args.emb_dim * 2, self.args.num_layers * self.args.emb_dim)

    def init_ent_emb_matrix(self, g):
        """ Initialize feature of entities by matrix form.

        Args:
            g: The dgl graph of meta task.
        """
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)

        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        
        if self.args.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.args.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.args.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)  # [B, self.inp_dim]

    def comp_ht_emb(self, head_embs, tail_embs):
        """combining embedding of head and tail.

        Args:
            head_embs: Embedding of heads.
            tail_embs: Embedding of tails.

        Returns:
            ht_embs: Embedding of head and tail.
        """
        if self.args.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.args.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.args.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.args.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_emb, tail_emb, rel_emb):
        """combining embedding of head, relation and tail.

        Args:
            head_emb: Embedding of head.
            relation_emb: Embedding of relation.
            tail_emb: Embedding of tail.

        Returns:
            hrt_embs: Embedding of head, relation and tail.
        """
        rel_emb = rel_emb.repeat(1, self.args.num_layers)
        if self.args.decoder_model.lower() == 'transe':
            hrt_embs = TransE.score_embedding(self, head_emb, rel_emb, tail_emb)
        elif self.args.decoder_model.lower() == 'distmult':
            hrt_embs = DistMult.score_embedding(self, head_emb, rel_emb, tail_emb)
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.args.decoder_model} not recognized.')
        
        return hrt_embs

    def nei_rel_path(self, g, rel_labels, r_emb_out):
        """Neighboring relational path module.

        Only consider in-degree relations first.

        Args:
            g: Subgraph of corresponding triple.
            rel_labels: Labels of relation.
            r_emb_out: Embedding of relation.

        Returns:
            output: Aggregate paths.
        """
        nei_rels = g.ndata['in_nei_rels']
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        heads_rels = nei_rels[head_ids]
        tails_rels = nei_rels[tail_ids]

        # Extract neighboring relational paths
        batch_paths = []
        for (head_rels, r_t, tail_rels) in zip(heads_rels, rel_labels, tails_rels):
            paths = []
            for h_r in head_rels:
                for t_r in tail_rels:
                    path = [h_r, r_t, t_r]
                    paths.append(path)
            batch_paths.append(paths)       # [B, n_paths, 3] , n_paths = n_head_rels * n_tail_rels
        
        batch_paths = torch.LongTensor(batch_paths).type_as(rel_labels)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        batch_size = batch_paths.shape[0]
        batch_paths = batch_paths.view(batch_size * len(paths), -1) # [B * n_paths, 3]

        batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # Input RNN 
        _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        last_state = last_state.squeeze(0) # squeeze the dim 0 
        last_state = last_state.view(batch_size, len(paths), self.args.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention
        if self.args.path_agg == 'mean':
            output = torch.mean(last_state, 1) # [B, inp_dim]
        
        if self.args.path_agg == 'att':
            r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
            atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
            atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
            output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        else:
            raise ValueError('unknown path_agg')
        
        return output # [B, inp_dim]

    def get_logits(self, s_G, s_g_pos, s_g_cor):
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret
    
    def forward(self, data, is_return_emb=False, cor_graph=False):
        """Getting the subgraph-level embedding.

        Args:
            data: Subgraphs and relation labels.
            is_return_emb: Whether return embedding.
            cor_graph: Whether corrupt the node feature.

        Returns:
            output: Representaion of subgraph.
            s_G: Global Subgraph embeddings.
            s_g: Local Subgraph embeddings.   
        """
        # Initialize the embedding of entities
        g, rel_labels = data
        g = dgl.batch(g)
        # Neighboring Relational Feature Module
        ## Initialize the embedding of nodes by neighbor relations
        if self.args.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)
        
        # Corrupt the node feature
        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]  
        
        # r: Embedding of relation
        r = self.rel_emb.weight.clone()
        
        # Input graph into GNN to get embeddings.
        g.ndata['h'], r_emb_out = self.gnn(g, r)
        
        # GRU layer for nodes
        graph_sizes = g.batch_num_nodes
        out_dim = self.args.num_layers * self.args.emb_dim
        g.ndata['repr'] = F.relu(self.batch_gru(g.ndata['repr'].view(-1, out_dim), graph_sizes()))
        node_hiddens = F.relu(self.W_o(g.ndata['repr']))  # num_nodes x hidden 
        g.ndata['repr'] = self.dropout(node_hiddens)  # num_nodes x hidden
        g_out = dgl.mean_nodes(g, 'repr').view(-1, out_dim)

        # Get embedding of target nodes (i.e. head and tail nodes)
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out,
                               head_embs.view(-1, out_dim),
                               tail_embs.view(-1, out_dim),
                               F.embedding(rel_labels, r_emb_out, padding_idx=-1)], dim=1)
        else:
            g_rep = torch.cat([g_out, self.rel_emb(rel_labels)], dim=1)
        
        # Represent subgraph by composing (h,r,t) in some way. (Not use in paper)
        if self.args.decoder_model:
            edge_embs = self.comp_hrt_emb(head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim), F.embedding(rel_labels, r_emb_out, padding_idx=-1))
            g_rep = torch.cat([g_out, edge_embs], dim=1)

        # Model neighboring relational paths 
        if self.args.nei_rel_path:
            # Model neighboring relational path
            g_p = self.nei_rel_path(g, rel_labels, r_emb_out)
            g_rep = torch.cat([g_rep, g_p], dim=1)
            s_g = torch.cat([g_out, g_p], dim=1)
        else:
            s_g = g_out
        output = self.fc_layer(g_rep)

        self.r_emb_out = r_emb_out
        
        if not is_return_emb:
            return output
        else:
            s_G = s_g.mean(0)
            return output, s_G, s_g

class RelCompGraphConv(RelGraphConv):
    """Basic layer of RGCN for SNRI.

    Attributes:
        args: Model configuration parameters.
        bias: Weight bias.
        inp_dim: Dimension of input.
        out_dim: Dimension of output.
        num_rels: The number of relations.
        num_bases: The number of bases.
        has_attn: Whether there is attention mechanism.
        is_input_layer: Whether it is input layer.
        aggregator: Type of aggregator.
        weight: Weight matrix.
        w_comp: Bases matrix.
        self_loop_weight: Self-loop weight.
        edge_dropout: Dropout of edge.
    """
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
        self.w_rel = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))
     
        self.edge_dropout = nn.Dropout(edge_dropout)

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None):
        """Message propagate function.

        Propagate messages and perform calculations according to the graph traversal order.

        Args:
            g: Subgraph of triple.
            attn_rel_emb: Relation attention embedding.
        """
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1)).type_as(weight)
        input_ = 'init' if self.is_input_layer else 'h'

        def comp(h, edge_data):
            """ Refer to CompGCN """
            if self.args.is_comp == 'mult':
                return h * edge_data
            elif self.args.is_comp == 'sub':
                return h - edge_data
            else:
                raise KeyError(f'composition operator {self.comp} not recognized.')

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            
            # Similar to CompGCN to interact nodes and relations
            if self.args.is_comp:
                edge_data = comp(edges.src[input_], F.embedding(edges.data['type'], self.rel_emb, padding_idx=-1))
            else:
                edge_data = edges.src[input_]

            msg = edges.data['w'] * torch.bmm(edge_data.unsqueeze(1), w).squeeze(1)

            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']), attn_rel_emb(edges.data['label'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).type_as(w)

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)

    def forward(self, g, rel_emb, attn_rel_emb=None):
        """Update node representation.

        Args:
            graph: Subgraph of corresponding triple.
            rel_emb: Embedding of relation.
            attn_rel_emb: Embedding of relation attention.

        Returns:
            rel_emb_out: Embedding of relation.
        """
        self.rel_emb = rel_emb
        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

        rel_emb_out = torch.matmul(self.rel_emb, self.w_rel)
        rel_emb_out[-1, :].zero_()       # padding embedding as 0
        
        return rel_emb_out

class Discriminator(nn.Module):
    """Discriminator module for calculating MI.

    Attributes:
        n_e: dimension of edge embedding.
        n_g: dimension of graph embedding.
    """
    def __init__(self, n_e, n_g):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_e, n_g, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """Init weights of layers.

        Args:
            m: Model layer.
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """For calculating MI loss.

        Attributes:
            c: Global Subgraph embeddings.
            h_pl: Positive local Subgraph embeddings.
            h_mi: Negative local Subgraph embeddings.
            s_bias1: Bias of sc_1.
            s_bias2: Bias of sc_2.
        """
        c_x = torch.unsqueeze(c, 0) # [1, F]
        c_x = c_x.expand_as(h_pl)   #[B, F]

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1) # [B];  self.f_k(h_pl, c_x): [B, 1]
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1) # [B]

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits