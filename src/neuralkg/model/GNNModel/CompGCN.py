import torch
from torch import nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from neuralkg.model import ConvE

class CompGCN(nn.Module):

    """`Composition-based multi-relational graph convolutional networks`_ (CompGCN), 
        which jointly embeds both nodes and relations in a relational graph.

    Attributes:
        args: Model configuration parameters.
    
    .. _Composition-based multi-relational graph convolutional networks: 
        https://arxiv.org/pdf/1911.03082.pdf
    """

    def __init__(self, args):
        super(CompGCN, self).__init__()
        self.args      = args
        self.ent_emb   = None
        self.rel_emb   = None 
        self.GraphCov  = None  

        self.init_model()

    def init_model(self):
        """Initialize the CompGCN model and embeddings 

        Args:
            ent_emb: Entity embedding, shape:[num_ent, emb_dim].
            rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
            GraphCov: The comp graph convolution layers.
            conv1: The convolution layer.
            fc: The full connection layer.
            bn0, bn1, bn2: The batch Normalization layer.
            inp_drop, hid_drop, feg_drop: The dropout layer.
        """
        #------------------------------CompGCN--------------------------------------------------------------------
        self.ent_emb = nn.Parameter(torch.Tensor(self.args.num_ent, self.args.emb_dim))
        self.rel_emb = nn.Parameter(torch.Tensor(self.args.num_rel, self.args.emb_dim))

        nn.init.xavier_normal_(self.ent_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.GraphCov = CompGCNCov(self.args.emb_dim, self.args.emb_dim * 2, torch.tanh, \
                                    bias = 'False', drop_rate = 0.1, opn = self.args.opn)
        
        self.bias = nn.Parameter(torch.zeros(self.args.num_ent))
        self.drop = nn.Dropout(0.3)
        #-----------------------------ConvE-----------------------------------------------------------------------
        self.emb_ent = torch.nn.Embedding(self.args.num_ent, self.args.emb_dim*2)
        self.inp_drop = torch.nn.Dropout(self.args.inp_drop)
        self.hid_drop = torch.nn.Dropout(self.args.hid_drop)
        self.feg_drop = torch.nn.Dropout2d(self.args.fet_drop)

        self.conv1 = torch.nn.Conv2d(1, 200, (7, 7), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(200)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.args.num_ent)))
        self.fc = torch.nn.Linear(39200, self.args.out_dim)

    def forward(self, graph, relation, norm, triples):
        """The functions used in the training phase

        Args:
            graph: The knowledge graph recorded in dgl.graph()
            relation: The relation id sampled in triples
            norm: The edge norm in graph 
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].

        Returns:
            score: The score of triples.
        """
        head, rela = triples[:,0], triples[:, 1]
        x, r = self.ent_emb, self.rel_emb  # embedding of relations
        x, r = self.GraphCov(graph, x, r, relation, norm)
        x = self.drop(x)  # embeddings of entities [num_ent, dim]
        head_emb = torch.index_select(x, 0, head)  # filter out embeddings of subjects in this batch
        #head_in_emb = head_emb.view(-1, 1, 10, 20)

        rela_emb = torch.index_select(r, 0, rela)  # filter out embeddings of relations in this batch
        #rela_in_emb = rela_emb.view(-1, 1, 10, 20)

        if self.args.decoder_model.lower() == 'conve':
           # score = ConvE.score_func(self, head_in_emb, rela_in_emb, x)
           score = self.ConvE(head_emb, rela_emb, x)

        elif self.args.decoder_model.lower() == 'distmult':
            score = self.DistMult(head_emb, rela_emb)
        
        else:
            raise ValueError("please choose decoder (DistMult/ConvE)")

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
        relation   = batch['rela']
        norm       = batch['norm'] 

        head, rela = triples[:,0], triples[:, 1]
        x, r = self.ent_emb, self.rel_emb  # embedding of relations
        x, r = self.GraphCov(graph, x, r, relation, norm)
        x = self.drop(x)  # embeddings of entities [num_ent, dim]
        head_emb = torch.index_select(x, 0, head)  # filter out embeddings of subjects in this batch
        #head_in_emb = head_emb.view(-1, 1, 10, 20)

        rela_emb = torch.index_select(r, 0, rela)  # filter out embeddings of relations in this batch
        #rela_in_emb = rela_emb.view(-1, 1, 10, 20)

        if self.args.decoder_model.lower() == 'conve':
           # score = ConvE.score_func(self, head_in_emb, rela_in_emb, x)
           score = self.ConvE(head_emb, rela_emb, x)

        elif self.args.decoder_model.lower() == 'distmult':
            score = self.DistMult(head_emb, rela_emb)
        
        else:
            raise ValueError("please choose decoder (DistMult/ConvE)")

        return score

    def DistMult(self, head_emb, rela_emb):
        """Calculating the score of triples with DistMult model."""
        obj_emb = head_emb * rela_emb  # [batch_size, emb_dim]
        x = torch.mm(obj_emb, self.emb_ent.weight.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score
    
    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, 200)
        rel_embed = rel_embed.view(-1, 1, 200)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        stack_input = stack_input.reshape(-1, 1, 2 * 10, 20)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def ConvE(self, sub_emb, rel_emb, all_ent):
        """Calculating the score of triples with ConvE model."""
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv1(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feg_drop(x)
        x = x.view(x.shape[0], -1)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

class CompGCNCov(nn.Module):
    """ The comp graph convolution layers, similar to https://github.com/malllabiisc/CompGCN"""
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr'):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn
        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.udf.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a.real, a.imag
            r2, i2 = b.real, b.imag
            real = r1 * r2 - i1 * i2
            imag = r1 * i2 + i1 * r2
            return torch.complex(real, imag)

        def conj(a):
            a.imag = -a.imag
            return a

        def ccorr(a, b):
            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), a.shape[-1])

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.graph, x, rel_repr, edge_type, edge_norm):
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)