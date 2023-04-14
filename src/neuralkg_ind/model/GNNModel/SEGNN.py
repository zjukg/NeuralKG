import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from neuralkg_ind import utils
from neuralkg_ind.utils.tools import get_param
from neuralkg_ind.model import ConvE

class SEGNN(nn.Module):
    def __init__(self, args):
        super(SEGNN, self).__init__() 
        self.device = torch.device("cuda:0") #TODO: remove cuda
        self.args = args
        self.dataset = self.args.dataset_name
        self.n_ent = self.args.num_ent
        self.n_rel = self.args.num_rel
        self.emb_dim = self.args.emb_dim

        # entity embedding
        self.ent_emb = get_param(self.n_ent, self.emb_dim)

        # gnn layer
        self.kg_n_layer = self.args.kg_layer  #1
        # relation SE layer
        self.edge_layers = nn.ModuleList([EdgeLayer(self.args) for _ in range(self.kg_n_layer)])
        # entity SE layer
        self.node_layers = nn.ModuleList([NodeLayer(self.args) for _ in range(self.kg_n_layer)])
        # triple SE layer
        self.comp_layers = nn.ModuleList([CompLayer(self.args) for _ in range(self.kg_n_layer)])

        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, self.emb_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        if self.args.pred_rel_w: #true
            self.rel_w = get_param(self.emb_dim * self.kg_n_layer, self.emb_dim).to(self.device)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, self.emb_dim)

        self.predictor = ConvE(self.args) #(200, 250, 7)
        self.ent_drop = nn.Dropout(self.args.ent_drop)  #0.2
        self.rel_drop = nn.Dropout(self.args.rel_drop)  #0
        self.ent_pred_drop = nn.Dropout(self.args.ent_drop_pred)
        self.act = nn.Tanh()

    def concat(self, head_emb, rela_emb):
        head_emb = head_emb.view(-1, 1, head_emb.shape[-1])
        rela_emb = rela_emb.view(-1, 1, rela_emb.shape[-1])
        stacked_input = torch.cat([head_emb, rela_emb], 1)
        stacked_input = torch.transpose(stacked_input, 2, 1).reshape((-1, 1, 2 * self.args.k_h, self.args.k_w))
        return stacked_input

    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        kg = kg.to(self.device)
        ent_emb, rel_emb = self.aggragate_emb(kg)
        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        # (bs, n_ent)
        ent_emb = self.ent_pred_drop(ent_emb)
        score = self.predictor.score_func(head, rel, self.concat, ent_emb)

        return score   

    def aggragate_emb(self, kg):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            ent_emb = ent_emb.to(self.device)
            rel_emb = rel_emb.to(self.device)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb
            rel_emb_list.append(rel_emb)

        if self.args.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1).to(self.device)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class CompLayer(nn.Module):
    def __init__(self, args):
        super(CompLayer, self).__init__()
        self.device = torch.device("cuda:0")
        self.args = args
        self.dataset = self.args.dataset_name
        self.n_ent = self.args.num_ent
        self.n_rel = self.args.num_rel
        self.emb_dim = self.args.emb_dim
        self.comp_op = self.args.comp_op    #'mul'
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(self.emb_dim, self.emb_dim).to(self.device)
        self.act = nn.Tanh()
        if self.args.bn:
            self.bn = torch.nn.BatchNorm1d(self.emb_dim).to(self.device)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel
        ent_emb = ent_emb.to(self.device)
        rel_emb = rel_emb.to(self.device)
        kg = kg.to(self.device)
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            # neihgbor entity and relation composition
            if self.args.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.args.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):
    def __init__(self, args):
        super(NodeLayer, self).__init__()
        self.device = torch.device("cuda:0")
        self.args = args
        self.dataset = self.args.dataset_name
        self.n_ent = self.args.num_ent
        self.n_rel = self.args.num_rel
        self.emb_dim = self.args.emb_dim

        self.neigh_w = get_param(self.emb_dim, self.emb_dim).to(self.device)
        self.act = nn.Tanh()
        if self.args.bn:
            self.bn = torch.nn.BatchNorm1d(self.emb_dim).to(self.device)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        kg = kg.to(self.device)
        ent_emb = ent_emb.to(self.device)
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # attention
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, args):
        super(EdgeLayer, self).__init__()
        self.device = torch.device("cuda:0")
        self.args = args
        self.dataset = self.args.dataset_name
        self.n_ent = self.args.num_ent
        self.n_rel = self.args.num_rel
        self.emb_dim = self.args.emb_dim

        self.neigh_w = utils.get_param(self.emb_dim, self.emb_dim).to(self.device)
        self.act = nn.Tanh()
        if self.args.bn:   # True
            self.bn = torch.nn.BatchNorm1d(self.emb_dim).to(self.device)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel
        kg = kg.to(self.device)
        ent_emb = ent_emb.to(self.device)
        rel_emb = rel_emb.to(self.device)
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # attention
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']
            
            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
