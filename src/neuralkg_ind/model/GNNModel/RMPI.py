import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class RMPI(nn.Module):
    """`Relational Message Passing for Fully Inductive Knowledge Graph Completion`_ (RMPI), which passes messages directly 
        between relations to make full use of the relation patterns for subgraph reasoning with new techniques on graph 
        transformation, graph pruning, relationaware neighborhood attention, addressing empty subgraphs, etc.

    Attributes:
        args: Model configuration parameters.
        rel_emb: Relation embedding, shape: [num_rel, rel_emb_dim].
        conc: Whether apply target-aware attention for 2-hop neighbors.

    .. _Relational Message Passing for Fully Inductive Knowledge Graph Completion: https://arxiv.org/abs/2210.03994
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.link_mode = 6
        self.is_big_dataset = False

        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.rel_emb_dim, sparse=False)

        torch.nn.init.normal_(self.rel_emb.weight)

        self.fc_reld1 = nn.ModuleList([nn.Linear(self.args.rel_emb_dim, self.args.rel_emb_dim, bias=True)
                                      for _ in range(6)
                                      ])
        self.fc_reld2 = nn.ModuleList([nn.Linear(self.args.rel_emb_dim, self.args.rel_emb_dim, bias=True)
                                      for _ in range(6)
                                      ])
        self.fc_reld = nn.Linear(self.args.rel_emb_dim, self.args.rel_emb_dim, bias=True)

        self.fc_layer = nn.Linear(self.args.rel_emb_dim, 1)

        if self.args.conc:
            self.conc = nn.Linear(self.args.rel_emb_dim*2, self.args.rel_emb_dim)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop = torch.nn.Dropout(self.args.edge_dropout)

    def AggregateConv(self, graph, u_node, v_node, num_nodes, num_edges, aggr_flag, is_drop):
        """Function of aggregating relation.

        Args:
            graph: Subgraph to corresponding triple.
            u_node: Node of head entities.
            v_node: Node of tail entities.
            num_nodes: The number of nodes.
            num_edges: The number of edges.
            agg_flag:  
                2: 2-hop neighbors
                1: 1-hop directed neighbors
                0: 1-hop disclosing directed neighbors
            drop: Whether mask edges.

        Returns:
            rel_neighbor_embd: embedding of relation neighbors.
        """
        u_in_edge = graph.in_edges(u_node, 'all')
        u_out_edge = graph.out_edges(u_node, 'all')
        v_in_edge = graph.in_edges(v_node, 'all')
        v_out_edge = graph.out_edges(v_node, 'all')

        edge_mask = self.drop(torch.ones(num_edges))
        edge_mask = edge_mask.repeat(num_nodes, 1)

        in_edge_out = torch.sparse_coo_tensor(torch.cat((u_in_edge[1].unsqueeze(0), u_in_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(u_in_edge[2])).type_as(u_node), size=torch.Size((num_nodes, num_edges)))
        out_edge_out = torch.sparse_coo_tensor(torch.cat((u_out_edge[0].unsqueeze(0), u_out_edge[2].unsqueeze(0)), 0),
                                               torch.ones(len(u_out_edge[2])).type_as(u_node), size=torch.Size((num_nodes, num_edges)))
        in_edge_in = torch.sparse_coo_tensor(torch.cat((v_in_edge[1].unsqueeze(0), v_in_edge[2].unsqueeze(0)), 0),
                                             torch.ones(len(v_in_edge[2])).type_as(u_node), size=torch.Size((num_nodes, num_edges)))
        out_edge_in = torch.sparse_coo_tensor(torch.cat((v_out_edge[0].unsqueeze(0), v_out_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(v_out_edge[2])).type_as(u_node), size=torch.Size((num_nodes, num_edges)))

        if is_drop:
            in_edge_out = self.sparse_dense_mul(in_edge_out, edge_mask)
            out_edge_out = self.sparse_dense_mul(out_edge_out, edge_mask)
            in_edge_in = self.sparse_dense_mul(in_edge_in, edge_mask)
            out_edge_in = self.sparse_dense_mul(out_edge_in, edge_mask)
        if self.is_big_dataset:  # smaller memory
            in_edge_out = self.sparse_index_select(in_edge_out, u_node)
            out_edge_out = self.sparse_index_select(out_edge_out, u_node)
            in_edge_in = self.sparse_index_select(in_edge_in, v_node)
            out_edge_in = self.sparse_index_select(out_edge_in, v_node)
        else:  # faster calculation
            in_edge_out = in_edge_out.to_dense()[u_node].to_sparse()
            out_edge_out = out_edge_out.to_dense()[u_node].to_sparse()
            in_edge_in = in_edge_in.to_dense()[v_node].to_sparse()
            out_edge_in = out_edge_in.to_dense()[v_node].to_sparse()

        edge_mode_5 = out_edge_out.mul(in_edge_in)
        edge_mode_6 = in_edge_out.mul(out_edge_in)
        out_edge_out = out_edge_out.sub(edge_mode_5)
        in_edge_in = in_edge_in.sub(edge_mode_5)
        in_edge_out = in_edge_out.sub(edge_mode_6)
        out_edge_in = out_edge_in.sub(edge_mode_6)

        if aggr_flag == 1:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                                                     self.fc_reld2[i](self.h1)) for i in range(self.link_mode)])

            return rel_neighbor_embd

        elif aggr_flag == 2:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            if self.args.target2nei_atten:
                xxx = self.rel_emb(self.neighbor_edges2rels)
                rel_2directed_atten = torch.einsum('bd,nd->bn', [xxx, self.h0])
                rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

                item = list()
                for i in range(6):
                    atten = self.sparse_dense_mul(edge_connect_l[i], rel_2directed_atten).to_dense()
                    mask = (atten == 0).bool()
                    atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
                    atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                                atten_softmax).to_sparse()
                    agg_i = torch.sparse.mm(atten_softmax, self.fc_reld1[i](self.h0))
                    item.append(agg_i)
                rel_neighbor_embd = sum(item)

            else:
                rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                                                         self.fc_reld1[i](self.h0)) for i in
                                         range(self.link_mode)])

            return rel_neighbor_embd

        elif aggr_flag == 0:
            num_target = u_node.shape[0]
            dis_target_edge_ids = self.rel_edge_ids
            self_mask = torch.ones((num_target, num_edges))
            for i in range(num_target):
                self_mask[i][dis_target_edge_ids[i]] = 0
            self_mask = self_mask
            edge_mode_5 = self.sparse_dense_mul(edge_mode_5, self_mask)


            edge_connect_l = in_edge_out + out_edge_out + in_edge_in + out_edge_in + edge_mode_5 + edge_mode_6

            neighbor_rel_embeds = self.rel_emb(graph.edata['type'])

            rel_2directed_atten = torch.einsum('bd,nd->bn', [self.fc_reld(self.rel_emb(self.rel_labels)), self.fc_reld(neighbor_rel_embeds)])
            rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

            atten = self.sparse_dense_mul(edge_connect_l, rel_2directed_atten).to_dense()
            mask = (atten == 0).bool()
            atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
            atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                            atten_softmax).to_sparse()
            rel_neighbor_embd = torch.sparse.mm(atten_softmax, self.fc_reld(neighbor_rel_embeds))

            return rel_neighbor_embd

    def forward(self, data):
        """calculating subgraphs score.

        Args:
            data: Enclosing/disclosing subgraphs and relation labels.

        Returns:
            output: socore of subgraphs.
        """
        (en_g, dis_g), rel_labels = data

        # relational aggregation begin
        self.rel_labels = rel_labels
        num_nodes = en_g.number_of_nodes()
        num_edges = en_g.number_of_edges()

        head_ids = (en_g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (en_g.ndata['id'] == 2).nonzero().squeeze(1)

        head_node, tail_node = head_ids, tail_ids
        u_in_nei = en_g.in_edges(head_node, 'all')
        u_out_nei = en_g.out_edges(head_node, 'all')
        v_in_nei = en_g.in_edges(tail_node, 'all')
        v_out_nei = en_g.out_edges(tail_node, 'all')

        edge2rel = dict()
        for i in range(len(rel_labels)):
            u_node_i = head_node[i]
            v_node_i = tail_node[i]
            u_i_in_edge = en_g.in_edges(u_node_i, 'all')[2]
            u_i_out_edge = en_g.out_edges(u_node_i, 'all')[2]
            v_i_in_edge = en_g.in_edges(v_node_i, 'all')[2]
            v_i_out_edge = en_g.out_edges(v_node_i, 'all')[2]
            i_neighbor_edges = torch.cat((u_i_in_edge, u_i_out_edge, v_i_in_edge, v_i_out_edge))
            i_neighbor_edges = torch.unique(i_neighbor_edges, sorted=False)
            # print(i_neighbor_edges)
            for eid in i_neighbor_edges.cpu().numpy().tolist():
                edge2rel[eid] = rel_labels[i]

        self.h0 = self.rel_emb(en_g.edata['type'])

        neighbor_edges = torch.cat((u_in_nei[2], u_out_nei[2], v_in_nei[2], v_out_nei[2]))
        neighbor_edges = torch.unique(neighbor_edges, sorted=False)

        neighbor_edges2rels = [edge2rel[eid] for eid in neighbor_edges.cpu().numpy().tolist()]
        neighbor_edges2rels = torch.Tensor(neighbor_edges2rels).type_as(self.h0).long()

        neighbor_u_nodes = en_g.edges()[0][neighbor_edges]
        neighbor_v_nodes = en_g.edges()[1][neighbor_edges]

        self.neighbor_edges = neighbor_edges
        self.neighbor_edges2rels = neighbor_edges2rels

        self.h0_extracted = self.h0[neighbor_edges]
        h_0_N = self.AggregateConv(en_g, neighbor_u_nodes, neighbor_v_nodes, num_nodes, num_edges, aggr_flag=2, is_drop=True)
        h_0_N = F.relu(h_0_N)
        self.h1 = self.rel_emb(en_g.edata['type'])

        for i, eid in enumerate(neighbor_edges):
            self.h1[eid] = self.h1[eid] + h_0_N[i]

        rel_edge_ids = torch.LongTensor([en_g.edge_id(head_ids[i], tail_ids[i]) for i in range(head_ids.shape[0])])

        self.h1_extracted = self.h1[rel_edge_ids]
        self.rel_edge_ids = rel_edge_ids
        self.rel_edge_ids = rel_edge_ids
        h_1_N = self.AggregateConv(en_g, head_node, tail_node, num_nodes, num_edges, aggr_flag=1, is_drop=True)

        h_1_N = F.relu(h_1_N)
        h2 = self.h1_extracted+h_1_N

        if self.args.ablation == 0: # RMP base
            final_embed = h2
            g_rep = F.normalize(final_embed, p=2, dim=-1)
        elif self.args.ablation == 1:  # RMP NE
            dis_head_ids = (dis_g.ndata['id'] == 1).nonzero().squeeze(1)
            dis_tail_ids = (dis_g.ndata['id'] == 2).nonzero().squeeze(1)

            dis_num_nodes = dis_g.number_of_nodes()
            dis_num_edges = dis_g.number_of_edges()
            one_hop_nei_embd = self.AggregateConv(dis_g, dis_head_ids, dis_tail_ids, dis_num_nodes, dis_num_edges,
                                             aggr_flag=0, is_drop=True)
            one_hop_nei_embd = F.relu(one_hop_nei_embd)
            if self.args.conc:
                h2 = F.normalize(h2, p=2, dim=-1)
                one_hop_nei_embd = F.normalize(one_hop_nei_embd, p=2, dim=-1)
                g_rep = self.conc(torch.cat([h2, one_hop_nei_embd], dim=1))
            else:
                final_embed = h2 + one_hop_nei_embd
                g_rep = F.normalize(final_embed, p=2, dim=-1)

        output = self.fc_layer(g_rep)
        return output

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]].to(device=torch.device("cuda:0"))  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        indices_s = s._indices()
        indice_new_1 = torch.tensor([])
        indice_new_2 = torch.tensor([])
        num_i = 0.0
        for itm in idx:
            mask = (indices_s[0] == itm)
            indice_tmp_1 = torch.ones(sum(mask)) * num_i
            indice_tmp_2 = indices_s[1][mask].float()
            indice_new_1 = torch.cat((indice_new_1, indice_tmp_1), dim=0)
            indice_new_2 = torch.cat((indice_new_2, indice_tmp_2), dim=0)
            num_i = num_i + 1.0
        indices_new = torch.cat((indice_new_1.unsqueeze(0), indice_new_2.unsqueeze(0)), dim=0).long()

        return torch.sparse.FloatTensor(indices_new, torch.ones(indices_new.shape[1]),
                                        torch.Size((len(idx), s.shape[1])))
