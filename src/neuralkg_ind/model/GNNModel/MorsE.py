import torch.nn as nn
import torch
import dgl
import pickle
import numpy as np
import torch.nn.functional as F
from .RGCN import RelGraphConv
from .model import Model
from neuralkg_ind.utils.tools import *
from neuralkg_ind.model import TransE, DistMult, ComplEx, RotatE
from neuralkg_ind.utils import get_indtest_test_dataset_and_train_g, get_g_bidir

class MorsE(nn.Module):
    """`Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding`_ (MorsE), which learns transferable meta-knowledge that
        can be used to produce entity embeddings.

    Attributes:
        args: Model configuration parameters.
        ent_init: Relation embedding init class.
        rgcn: RGCN model.
        KGEModel: KGE model.

    .. _Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding: https://arxiv.org/abs/2110.14170
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.ent_dim = args.emb_dim
        args.rel_dim = args.emb_dim
        if args.kge_model in ['ComplEx', 'RotatE']:
            args.ent_dim = args.emb_dim * 2
        if args.kge_model in ['ComplEx']:
            args.rel_dim = args.emb_dim * 2

        args.num_rel = self.get_num_rel(args)
        self.ent_init = EntInit(args)
        self.rgcn = RGCN(args, basiclayer = RelMorsGraphConv)
        self.kge_model = KGEModel(args)

    def forward(self, sample, ent_emb, mode='single'):
        """Calculating triple score.

        Args:
            sample: Sampled triplets.
            ent_emb: Embedding of entities.
            mode: This arg indicates that negative entity will replace the head or tail entity.

        Returns:
            score: Score of triple.
        """
        return self.kge_model(sample, ent_emb, mode)

    def get_intest_train_g(self):
        """Getting inductive test-train graph.

        Returns:
            indtest_train_g: test-train graph.
        """
        data, _, _ , _ = get_indtest_test_dataset_and_train_g(self.args)
        self.indtest_train_g = get_g_bidir(torch.LongTensor(data['train']), self.args)
        self.indtest_train_g = self.indtest_train_g.to(self.args.gpu)
        return self.indtest_train_g

    def get_ent_emb(self, sup_g_bidir):
        """Getting entities embedding.

        Args:
            sup_g_bidir: Undirected supporting graph.

        Returns:
            ent_emb: Embedding of entities.
        """
        self.ent_init(sup_g_bidir)
        ent_emb = self.rgcn(sup_g_bidir)

        return ent_emb
    
    def get_score(self, batch, mode):
        """Getting score of triplets.

        Args:
            batch: Including positive sample, entities embedding, etc.

        Returns:
            score: Score of positive or negative sample.
        """
        pos_triple = batch["positive_sample"]
        ent_emb = batch["ent_emb"]

        if batch['cand'] == 'all':
            return self.kge_model((pos_triple, None), ent_emb, mode)
        else:
            if mode == 'tail_predict':
                tail_cand = batch['tail_cand']
                return self.kge_model((pos_triple, tail_cand), ent_emb, mode)
            else:
                head_cand = batch['head_cand']
                return self.kge_model((pos_triple, head_cand), ent_emb, mode)
    
    def get_num_rel(self, args):
        """Getting number of relation.

        Args:
            args: Model configuration parameters.

        Returns:
            num_rel: The number of relation.
        """
        data = pickle.load(open(args.pk_path, 'rb'))
        num_rel = len(np.unique(np.array(data['train_graph']['train'])[:, 1]))

        return num_rel   

class EntInit(nn.Module):
    """Class of initializing entities.

    Attributes:
        args: Model configuration parameters.
        rel_head_emb: Embedding of relation to head.
        rel_tail_emb: Embedding of relation to tail.
    """
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, g_bidir):
        """Initialize entities in graph.

        Args:
            g_bidir: Undirected graph.
        """
        num_edge = g_bidir.num_edges()
        etypes = g_bidir.edata['type']
        g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.ent_dim).type_as(etypes).float()
        rh_idx = etypes < self.args.num_rel
        rt_idx = etypes >= self.args.num_rel
        g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[etypes[rh_idx]]
        g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[etypes[rt_idx] - self.args.num_rel]

        message_func = dgl.function.copy_e('ent_e', 'msg')
        reduce_func = dgl.function.mean('msg', 'feat')
        g_bidir.update_all(message_func, reduce_func)
        g_bidir.edata.pop('ent_e')

class RelMorsGraphConv(RelGraphConv):
    """Basic layer of RGCN.

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
    def __init__(self, args, inp_dim, out_dim, aggregator, num_rels, num_bases=-1, bias=False,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False):
        super().__init__(args, inp_dim, out_dim, 0, None, 0, bias=bias, activation=activation, 
                        self_loop=True, dropout=0.0, layer_norm=False)
        self.in_dim = inp_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.aggregator = aggregator
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels

        self.rel_weight = None
        self.input_ = None

        self.activation = activation

        self.is_input_layer = is_input_layer

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels*2, self.num_bases))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

        self.aggregator = self.aggregator

    def message(self, edges):
        """Message function for propagating.

        Args:
            edges: Edges in graph.

        Returns:
            curr_emb: Embedding of current layer.
            msg: Message for propagating.
            a: Coefficient.
        """
        w = self.rel_weight.index_select(0, edges.data['type'])
        msg = torch.bmm(edges.src[self.input_].unsqueeze(1), w).squeeze(1)
        curr_emb = torch.mm(edges.dst[self.input_], self.loop_weight)  # (B, F)
        a = 1 / edges.dst['in_d'].type_as(w).to(torch.float32).reshape(-1, 1)

        return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

    def apply_node_func(self, nodes):
        """Function used for nodes.

        Args:
            nodes: nodes in graph.

        Returns:
            node_repr: Representation of nodes.
        """
        node_repr = nodes.data['h']

        if self.bias:
            node_repr = node_repr + self.h_bias

        if self.activation:
            node_repr = self.activation(node_repr)

        return {'h': node_repr}

    def forward(self, g):
        """Update node representation.

        Args:
            g: Subgraph of corresponding triple.
        """
        # generate all relations' weight from bases
        weight = self.weight.view(self.num_bases, self.in_dim * self.out_dim)
        self.rel_weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels*2, self.in_dim, self.out_dim)

        # normalization constant
        g.dstdata['in_d'] = g.in_degrees()

        self.input_ = 'feat' if self.is_input_layer else 'h'

        g.update_all(self.message, self.aggregator, self.apply_node_func)

        if self.is_input_layer:
            g.ndata['repr'] = torch.cat([g.ndata['feat'], g.ndata['h']], dim=1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h']], dim=1)

class RGCN(Model):
    """RGCN model

    Attributes:
        args: Model configuration parameters.
        basiclayer: Layer of RGCN model.
        inp_dim: Dimension of input.
        emb_dim: Dimension of embedding.
        aggregator: Type of aggregator.
    """
    def __init__(self, args, basiclayer):
        super(RGCN, self).__init__(args)

        self.args = args
        self.basiclayer = basiclayer
        self.inp_dim = args.ent_dim
        self.emb_dim = args.ent_dim
        self.num_rel = args.num_rel
        self.num_bases = args.num_bases

        aggregator_type = self.args.gnn_agg_type.upper()+"Aggregator"
        aggregator_class = import_class(f"neuralkg_ind.model.{aggregator_type}")
        self.aggregator = aggregator_class(self.emb_dim)

        self.build_model()

        self.jk_linear = nn.Linear(self.emb_dim*(self.args.num_layers+1), self.emb_dim)

    def build_hidden_layer(self, idx): 
        """build hidden layer of RGCN.

        Args:
            idx: The idx of layer.

        Returns:
            output: Build a basic layer according to whether it is the first layer.
        """
        input_flag = True if idx == 0 else False
        return self.basiclayer(self.args, self.inp_dim, self.emb_dim, self.aggregator, self.num_rel, self.num_bases, bias=True,
                        activation=F.relu, is_input_layer=input_flag)

    def forward(self, g):
        """Getting nodes embedding.

        Args:
            g: Subgraph of corresponding task.

        Returns:
            g.ndata['h']: Nodes embedding.
        """
        for layer in self.layers:
            layer(g)

        g.ndata['h'] = self.jk_linear(g.ndata['repr'])
        return g.ndata['h']

class KGEModel(nn.Module):
    """KGE model

    Attributes:
        args: Model configuration parameters.
        model_name: The name of model.
        nrelation: The number of relation.
        emb_dim: Dimension of embedding.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        relation_embedding: Embedding of relation.
    """
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.model_name = args.kge_model
        self.nrelation = args.num_rel
        self.emb_dim = args.emb_dim
        self.epsilon = 2.0

        self.margin = torch.Tensor([args.margin])

        self.embedding_range = torch.Tensor([(self.margin.item() + self.epsilon) / args.emb_dim])
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.args.rel_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % self.model_name)

    def forward(self, sample, ent_emb, mode='single'):
        '''Forward function that calculate the score of a batch of triples.
            In the 'single' mode, sample is a batch of triple.
            In the 'head-batch' or 'tail-batch' mode, sample consists two part.
            The first part is usually the positive sample.
            And the second part is the entities in the negative samples.
            Because negative samples and positive samples usually share two elements
            in their triple ((head, relation) or (relation, tail)).
        
        Args:
            sample: Positive and negative sample.
            ent_emb: Embedding of entities.
            mode: 'single', 'head-batch' or 'tail-batch'.

        Returns:
            score: The score of sample.
        '''
        self.entity_embedding = ent_emb
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head_predict':
            tail_part, head_part = sample
            if head_part != None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if head_part == None:
                head = self.entity_embedding.unsqueeze(0)
            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail_predict':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        elif mode == 'rel-batch':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            if tail_part == None:
                relation = self.relation_embedding.unsqueeze(0)
            else:
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': TransE.score_func,
            'DistMult': DistMult.score_func,
            'ComplEx': ComplEx.score_func,
            'RotatE': RotatE.score_func,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](self, head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score