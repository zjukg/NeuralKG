import torch.nn as nn
import torch
import dgl
import pickle
import numpy as np
import torch.nn.functional as F
from neuralkg.model import TransE, DistMult, ComplEx, RotatE
from neuralkg.utils import get_indtest_test_dataset_and_train_g, get_g_bidir

class MorsE(nn.Module):
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
        self.rgcn = RGCN(args)
        self.kge_model = KGEModel(args)

        # data, _, _ , _ = get_indtest_test_dataset_and_train_g(args)
        # self.indtest_train_g = get_g_bidir(torch.LongTensor(data['train']), args)
        # self.indtest_train_g = self.indtest_train_g.to(self.args.gpu)

    def forward(self, sample, ent_emb, mode='single'):
        return self.kge_model(sample, ent_emb, mode)

    def get_intest_train_g(self):
        data, _, _ , _ = get_indtest_test_dataset_and_train_g(self.args)
        self.indtest_train_g = get_g_bidir(torch.LongTensor(data['train']), self.args)
        self.indtest_train_g = self.indtest_train_g.to(self.args.gpu)
        return self.indtest_train_g

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        ent_emb = self.rgcn(sup_g_bidir)

        return ent_emb
    
    def get_score(self, batch, mode):
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
        data = pickle.load(open(args.pk_path, 'rb'))
        num_rel = len(np.unique(np.array(data['train_graph']['train'])[:, 1]))

        return num_rel   

class EntInit(nn.Module):
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        
        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, g_bidir):
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=None, has_bias=False, activation=None,
                 is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels

        # for msg_func
        self.rel_weight = None
        self.input_ = None

        self.has_bias = has_bias
        self.activation = activation

        self.is_input_layer = is_input_layer

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels*2, self.num_bases))
        self.self_loop_weight = nn.Parameter(torch.Tensor(self.in_dim, self.out_dim))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.aggregator = Aggregator()

        # bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_dim))
            nn.init.zeros_(self.bias)

    def msg_func(self, edges):
        w = self.rel_weight.index_select(0, edges.data['type'])
        msg = torch.bmm(edges.src[self.input_].unsqueeze(1), w).squeeze(1)
        curr_emb = torch.mm(edges.dst[self.input_], self.self_loop_weight)  # (B, F)
        a = 1 / edges.dst['in_d'].to(torch.float32).type_as(w).reshape(-1, 1)

        return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

    def apply_node_func(self, nodes):
        node_repr = nodes.data['h']

        if self.has_bias:
            node_repr = node_repr + self.bias

        if self.activation:
            node_repr = self.activation(node_repr)

        return {'h': node_repr}

    def forward(self, g):
        # generate all relations' weight from bases
        weight = self.weight.view(self.num_bases, self.in_dim * self.out_dim)
        self.rel_weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels*2, self.in_dim, self.out_dim)

        # normalization constant
        g.dstdata['in_d'] = g.in_degrees()

        self.input_ = 'feat' if self.is_input_layer else 'h'

        g.update_all(self.msg_func, self.aggregator, self.apply_node_func)


        if self.is_input_layer:
            g.ndata['repr'] = torch.cat([g.ndata['feat'], g.ndata['h']], dim=1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h']], dim=1)


class RGCN(nn.Module):

    def __init__(self, args):
        super(RGCN, self).__init__()

        self.emb_dim = args.ent_dim
        self.num_rel = args.num_rel
        self.num_bases = args.num_bases
        self.num_layers = args.num_layers
        self.device = args.gpu

        # create rgcn layers
        self.layers = nn.ModuleList()
        self.build_model()

        self.jk_linear = nn.Linear(self.emb_dim*(self.num_layers+1), self.emb_dim)

    def build_model(self):
        # i2h
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # h2h
        for idx in range(self.num_layers - 1):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         self.num_rel,
                         self.num_bases,
                         has_bias=True,
                         activation=F.relu,
                         is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         self.num_rel,
                         self.num_bases,
                         has_bias=True,
                         activation=F.relu)

    def forward(self, g):

        for idx, layer in enumerate(self.layers):
            layer(g)

        g.ndata['h'] = self.jk_linear(g.ndata['repr'])
        return g.ndata['h']


class KGEModel(nn.Module):
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
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
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
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.margin.item() - score.sum(dim=2)
        return score