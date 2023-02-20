import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable    

import numpy as np
from .layer import BatchGRU

class CoMPILE(nn.Module):
    """`Communicative Message Passing for Inductive Relation Reasoning`_ (CoMPILE), which reasons over 
        local directed subgraph structures and strengthens the message interactions between edges and 
        entitles through a communicative kernel.

    Attributes:
        args: Model configuration parameters.
        latent_dim: Latent dimension.
        output_dim: Output dimension.
        node_emb: Dimension of node embedding.
        relation_emb: Dimension of relation embedding.
        hidden_size: Size of hidden layer.

    .. _Communicative Message Passing for Inductive Relation Reasoning: https://arxiv.org/pdf/2012.08911
    """
    def __init__(self, args):
        super(CoMPILE, self).__init__()  
        self.args = args
        self.latent_dim = self.args.emb_dim
        self.output_dim = 1
        self.node_emb = self.args.inp_dim
        self.relation_emb = self.args.rel_emb_dim
        self.edge_emb = self.node_emb * 2 + self.relation_emb 
        self.hidden_size = self.args.emb_dim

        self.final_relation_embeddings = nn.Parameter(torch.randn(self.args.aug_num_rels, self.args.rel_emb_dim))
        self.relation_to_edge = nn.Linear(self.args.rel_emb_dim, self.hidden_size)

        self.linear1 = nn.Linear(self.args.emb_dim , 16)
        self.linear2 = nn.Linear(16, 1)

        self.node_fdim = self.node_emb
        self.edge_fdim = self.edge_emb
        
        self.bias = False
        self.depth = 3
        self.dropout = 0.5
        self.layers_per_message = 1
        self.undirected = False
        self.node_messages = False

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.act_func = nn.ReLU()

        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size),  requires_grad=False)
        # Input
        input_dim = self.node_fdim
        self.W_i_node = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.edge_fdim
        self.W_i_edge = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size_node = self.hidden_size + self.edge_fdim
        self.W_h_node = nn.Linear(w_h_input_size_node, self.hidden_size, bias=self.bias)

        self.input_attention1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=self.bias)
        self.input_attention2 = nn.Linear(self.hidden_size, 1, bias=self.bias)
        
        w_h_input_size_edge = self.hidden_size
        for depth in range(self.depth-1):
            self._modules['W_h_edge_{}'.format(depth)] = nn.Linear(w_h_input_size_edge, self.hidden_size, bias=self.bias)
            self._modules['Attention1_{}'.format(depth)] = nn.Linear(self.hidden_size + self.relation_emb, self.hidden_size, bias=self.bias)
            self._modules['Attention2_{}'.format(depth)] = nn.Linear(self.hidden_size, 1, bias=self.bias)
        
        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.gru = BatchGRU(self.hidden_size)
        
        self.communicate_mlp = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        
        for depth in range(self.depth-1):
            self._modules['W_h_node_{}'.format(depth)] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

    def forward(self, subgraph):
        """calculating subgraphs score.

        Args:
            subgraph: Subgraph of triple.

        Returns:
            out_conv: The output of convolution layer.
        """
        subgraph = subgraph[0]
        target_relation = []
        for i in range(len(subgraph)):
            graph = subgraph[i]
            target = graph.edata['label'][-1].squeeze()
            target_relation.append(self.final_relation_embeddings[target, :].unsqueeze(0))
        target_relation = torch.cat(target_relation, dim = 0)
        graph_embed, source_embed, target_embed = self.batch_subgraph(subgraph) 
        
        conv_input = torch.tanh(source_embed + target_relation -target_embed)   
        out_conv = (self.linear1(conv_input))
        out_conv = self.linear2(out_conv)
        return out_conv

    def batch_subgraph(self, subgraph):
        """calculating subgraphs score.

        Args:
            subgraph: Subgraph of triple.

        Returns:
            graph_embed: Embedding of subgraph.
            source_embed: Embedding of source entities.
            target_embed: Embedding of target entities.
        """
        graph_sizes = []; node_feat = []
        list_num_nodes = np.zeros((len(subgraph), ), dtype=np.int32)
        list_num_edges = np.zeros((len(subgraph), ), dtype=np.int32)
        node_count = 0 ; edge_count = 0; edge_feat = []
        total_edge = []; source_node = []; target_node = [] 
        total_target_relation = []; total_edge2 = []
        total_source = []; total_target = []
        for i in range(len(subgraph)):
            graph = subgraph[i]      
            node_embedding = graph.ndata['feat']
            node_feat.append(node_embedding)
            
            graph_sizes.append(graph.number_of_nodes())
            list_num_nodes[i] = graph.number_of_nodes()
            list_num_edges[i] = graph.number_of_edges()
 
            nodes = list((graph.nodes()).data.cpu().numpy())
            source = list((graph.edges()[0]).data.cpu().numpy()) 
            target = list((graph.edges()[1]).data.cpu().numpy())           
            relation = graph.edata['type']             
            relation_now = self.final_relation_embeddings[relation, :]
           
            target_relation = graph.edata['label']
            target_relation_now = self.final_relation_embeddings[target_relation, :]
            total_target_relation.append(target_relation_now)

            mapping = dict(zip(nodes, [i for i in range(node_count, node_count+list_num_nodes[i])]))

            source_map_now = np.array([mapping[v] for v in source]) - node_count
            target_map_now = np.array([mapping[v] for v in target]) - node_count
            source_embed = node_embedding[source_map_now, :]
            target_embed = node_embedding[target_map_now, :]

            edge_embed = torch.cat([source_embed, relation_now, target_embed], dim = 1)
            edge_feat.append(edge_embed)
            
            source_now = (graph.ndata['id'] == 1).nonzero().squeeze() + node_count
            target_now = (graph.ndata['id'] == 2).nonzero().squeeze() + node_count
            source_node.append(source_now)
            target_node.append(target_now)
            
            target_now = target_now.unsqueeze(0).repeat(list_num_edges[i], 1).long()
            source_now = source_now.unsqueeze(0).repeat(list_num_edges[i], 1).long()
            total_source.append(source_now); total_target.append(target_now)
            
            node_count += list_num_nodes[i]

            source_map = torch.LongTensor(np.array([mapping[v] for v in source])).unsqueeze(0)
            target_map = torch.LongTensor(np.array([mapping[v] for v in target])).unsqueeze(0)
          
            edge_pair = torch.cat([target_map, torch.LongTensor(np.array(range(edge_count, edge_count+list_num_edges[i]))).unsqueeze(0)], dim=0)
            
            edge_pair2 = torch.cat([source_map, torch.LongTensor(np.array(range(edge_count, edge_count+list_num_edges[i]))).unsqueeze(0)], dim=0)

            edge_count += list_num_edges[i]
            total_edge.append(edge_pair)       
            total_edge2.append(edge_pair2)      
  
        source_node = np.array(torch.tensor(source_node, device='cpu'))
        target_node = np.array(torch.tensor(target_node, device='cpu')) 

        total_edge = torch.cat(total_edge, dim = 1).type_as(self.final_relation_embeddings).long()
        total_edge2 = torch.cat(total_edge2, dim = 1).type_as(self.final_relation_embeddings).long()
        total_target_relation = torch.cat(total_target_relation, dim=0)
        total_source = torch.cat(total_source, dim=0)
        total_target = torch.cat(total_target, dim=0)

        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        e2n_value = torch.FloatTensor(torch.ones(total_edge.shape[1])).type_as(self.final_relation_embeddings)
        e2n_sp = torch.sparse.FloatTensor(total_edge, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
        e2n_sp2 = torch.sparse.FloatTensor(total_edge2, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
        
        node_feat = torch.cat(node_feat, dim=0)

        edge_feat = torch.cat(edge_feat, dim=0)
        graph_embed, source_embed, target_embed = self.gnn(node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes, 
        total_target_relation, total_source, total_target, source_node, target_node, list(list_num_edges))

        return graph_embed, source_embed, target_embed

    def gnn(self, node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes, target_relation, total_source, total_target, source_node, target_node, edge_sizes = None, node_degs=None):
        """calculating graph embedding, source embedding and target embedding.

        Args:
            node_feat: Feature of nodes.
            edge_feat: Feature of edges.
            e2n_sp: Sparse matrix of edges to source nodes.
            e2n_sp2: Sparse matrix of edges to target nodes.
            graph_sizes: The number of each graph nodes.
            target_relation: Target relation label.
            total_source: Total source nodes.
            total_target: Total target nodes.
            source_node: Source node of triple.
            target_node: Target node of triple.
            edge_sizes: The sizes of edges.
            node_degs: The degrees of nodes.

        Returns:
            gmol_vecs: Graph embedding.
            source_embed: source node embedding.
            target_embed: target node embedding. 
        """
        input_node = self.W_i_node(node_feat)  # num_nodes x hidden_size
        input_node = self.act_func(input_node)
        message_node = input_node.clone()
        relation_embed = (edge_feat[:, self.node_emb: self.node_emb + self.relation_emb])
        
        
        input_edge = self.W_i_edge(edge_feat)  # num_edges x hidden_size
        message_edge = self.act_func(input_edge)
        input_edge = self.act_func(input_edge)

        graph_source_embed = message_node[total_source, :].squeeze(1)
        graph_target_embed = message_node[total_target, :].squeeze(1)
        graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
        edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
        edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
        edge_message = edge_source_message + relation_embed - edge_target_message

        attention = torch.cat([graph_edge_embed, edge_message], dim=1)
        attention = torch.relu(self.input_attention1(attention))
        attention = torch.sigmoid(self.input_attention2(attention))
        
        
        # Message passing
        for depth in range(self.depth - 1):
            message_edge = (message_edge * attention)
            agg_message = gnn_spmm(e2n_sp, message_edge)
            message_node = message_node + agg_message
            message_node = self.act_func(self._modules['W_h_node_{}'.format(depth)](message_node))

            edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
            edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
            message_edge = torch.relu(message_edge + torch.tanh( edge_source_message + relation_embed - edge_target_message))
            message_edge = self._modules['W_h_edge_{}'.format(depth)](message_edge)
            message_edge = self.act_func(input_edge + message_edge)
            message_edge = self.dropout_layer(message_edge)  # num_edges x hidden

            graph_source_embed = message_node[total_source, :].squeeze(1)
            graph_target_embed = message_node[total_target, :].squeeze(1)
            graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
            edge_message = edge_source_message + relation_embed - edge_target_message
            attention = torch.cat([graph_edge_embed, edge_message], dim=1)            
            attention = torch.relu(self._modules['Attention1_{}'.format(depth)](attention))
            attention = torch.sigmoid(self._modules['Attention2_{}'.format(depth)](attention))

        message_edge = (message_edge * attention)
        agg_message = gnn_spmm(e2n_sp, message_edge)
        agg_message2 = self.communicate_mlp(torch.cat([agg_message, message_node, input_node], 1))

        a_message = torch.relu(self.gru(agg_message2, graph_sizes))        
        node_hiddens = self.act_func(self.W_o(a_message))  # num_nodes x hidden
        node_hiddens = self.dropout_layer(node_hiddens)  # num_nodes x hidden
        
        # Readout
        mol_vecs = []
        a_start = 0        
        for a_size in graph_sizes:
            if a_size == 0:
                assert 0
            cur_hiddens = node_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
            a_start += a_size
        mol_vecs = torch.stack(mol_vecs, dim=0)    
        
        source_embed = node_hiddens[source_node, :]
        target_embed = node_hiddens[target_node, :]

        return mol_vecs, source_embed, target_embed         

class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)    
