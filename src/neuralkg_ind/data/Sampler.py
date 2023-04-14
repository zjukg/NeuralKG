import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import random
from .DataPreprocess import *
import dgl 
import torch.nn.functional as F
import time
import queue
from neuralkg_ind.utils.tools import subgraph_extraction_labeling
import math

class SubSampler(BaseGraph):
    """Sampling subgraphs. 
    
    Prepare subgraphs and collect batch of subgraphs. 
    """
    def __init__(self, args):
        super().__init__(args)

    def sampling(self, data):
        """Sampling function to collect batch of subgraph for training.
        
        Args:
            data: List of train data.
        
        Returns:
            batch_data: batch of train data.
        """
        batch_data = {}

        graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))

        graphs_neg = [item for sublist in graphs_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
        
        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = graphs_pos
        batch_data["negative_sample"] = graphs_neg
        batch_data["positive_label"] =  r_labels_pos
        batch_data["negative_label"] =  r_labels_neg
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'positive_label', 'negative_label']

class RMPISampler(BaseGraph): 
    """Sampling subgraphs for RMPI training, which add disclosing subgraph.
    """
    def __init__(self, args):
        super().__init__(args)

    def sampling(self, data):
        """Sampling function to collect batch of subgraph for RMPI training.
        
        Args:
            data: List of RMPI train data.
        
        Returns:
            batch_data: batch of RMPI train data.
        """
        batch_data = {}

        en_graphs_pos, dis_graphs_pos, g_labels_pos, r_labels_pos, en_graphs_negs, dis_graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))
        batched_en_graph_pos = dgl.batch(en_graphs_pos)
        batched_dis_graph_pos = dgl.batch(dis_graphs_pos)

        en_graphs_neg = [item for sublist in en_graphs_negs for item in sublist]
        dis_graphs_neg = [item for sublist in dis_graphs_negs for item in sublist]
        g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

        batched_en_graph_neg = dgl.batch(en_graphs_neg)
        batched_dis_graph_neg = dgl.batch(dis_graphs_neg)
        
        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = (batched_en_graph_pos, batched_dis_graph_pos)
        batch_data["negative_sample"] = (batched_en_graph_neg, batched_dis_graph_neg)
        batch_data["positive_label"] =  r_labels_pos
        batch_data["negative_label"] =  r_labels_neg
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'positive_label', 'negative_label']

class UniSampler(BaseSampler):
    """Random negative sampling 
    Filtering out positive samples and selecting some samples randomly as negative samples.

    Attributes:
        cross_sampling_flag: The flag of cross sampling head and tail negative samples.
    """
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.
        
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data
    
    def uni_sampling(self, data):
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.arrary(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.arrary(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']

class BernSampler(BaseSampler):
    """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
    Attributes:
        lef_mean: Record the mean of head entity
        rig_mean: Record the mean of tail entity
    """
    def __init__(self, args):
        super().__init__(args)
        self.lef_mean, self.rig_mean = self.calc_bern()
    def __normal_batch(self, h, r, t, neg_size):
        """Generate replace head/tail list according to Bernoulli distribution.
        
        Args:
            h: The head of triples.
            r: The relation of triples.
            t: The tail of triples.
            neg_size: The number of negative samples corresponding to each triple

        Returns:
             numpy.array: replace head list and replace tail list.
        """
        neg_size_h = 0
        neg_size_t = 0
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r])
        for i in range(neg_size):
            if random.random() > prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        res = []

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
        
        for hh in neg_list_h[:neg_size_h]:
            res.append((hh, r, t))
        
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)
        
        for tt in neg_list_t[:neg_size_t]:
            res.append((h, r, tt))

        return res

    def sampling(self, data):
        """Using bernoulli distribution to select whether to replace the head entity or tail entity.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []

        batch_data['mode'] = 'bern'
        for h, r, t in data:
            neg_ent = self.__normal_batch(h, r, t, self.args.num_neg)
            neg_ent_sample += neg_ent
        
        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data["negative_sample"] = torch.LongTensor(np.array(neg_ent_sample))

        return batch_data
    
    def calc_bern(self):
        """Calculating the lef_mean and rig_mean.
        
        Returns:
            lef_mean: Record the mean of head entity.
            rig_mean: Record the mean of tail entity.
        """
        h_of_r = ddict(set)
        t_of_r = ddict(set)
        freqRel = ddict(float)
        lef_mean = ddict(float)
        rig_mean = ddict(float)
        for h, r, t in self.train_triples:
            freqRel[r] += 1.0
            h_of_r[r].add(h)
            t_of_r[r].add(t)
        for r in h_of_r:
            lef_mean[r] = freqRel[r] / len(h_of_r[r])
            rig_mean[r] = freqRel[r] / len(t_of_r[r])
        return lef_mean, rig_mean

    @staticmethod
    def sampling_keys():
        return ['positive_sample', 'negative_sample', 'mode']

class AdvSampler(BaseSampler):
    """Self-adversarial negative sampling, in math:
    
    p\left(h_{j}^{\prime}, r, t_{j}^{\prime} \mid\left\{\left(h_{i}, r_{i}, t_{i}\right)\right\}\right)=\frac{\exp \alpha f_{r}\left(\mathbf{h}_{j}^{\prime}, \mathbf{t}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)}
    
    Attributes:
        freq_hr: The count of (h, r) pairs.
        freq_tr: The count of (t, r) pairs.
    """
    def __init__(self, args):
        super().__init__(args)
        self.freq_hr, self.freq_tr = self.calc_freq()
    def sampling(self, pos_sample):
        """Self-adversarial negative sampling.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        data = pos_sample.numpy().tolist()
        adv_sampling = []
        for h, r, t in data:
            weight = self.freq_hr[(h, r)] + self.freq_tr[(t, r)]
            adv_sampling.append(weight)
        adv_sampling = torch.tensor(adv_sampling, dtype=torch.float32).cuda()
        adv_sampling = torch.sqrt(1 / adv_sampling)
        return adv_sampling
    def calc_freq(self):
        """Calculating the freq_hr and freq_tr.
        
        Returns:
            freq_hr: The count of (h, r) pairs.
            freq_tr: The count of (t, r) pairs.
        """
        freq_hr, freq_tr = {}, {}
        for h, r, t in self.train_triples:
            if (h, r) not in freq_hr:
                freq_hr[(h, r)] = self.args.freq_init
            else:
                freq_hr[(h, r)] += 1
            if (t, r) not in freq_tr:
                freq_tr[(t, r)] = self.args.freq_init
            else:
                freq_tr[(t, r)] += 1
        return freq_hr, freq_tr

class AllSampler(RevSampler):
    """Merging triples which have same head and relation, all false tail entities are taken as negative samples.    
    """
    def __init__(self, args):
        super().__init__(args)
        # self.num_rel_without_rev = self.args.num_rel // 2
        
    def sampling(self, data):
        """Randomly sampling from the merged triples.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        # sample_id = [] #确定triple里的relation是否是reverse的。reverse为1，不是为0
        batch_data = {}
        table = torch.zeros(len(data), self.args.num_ent)
        for id, (h, r, _) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            table[id][hr_sample] = 1
            # if r > self.num_rel_without_rev:
            #     sample_id.append(1)
            # else:
            #     sample_id.append(0)
        batch_data["sample"] = torch.LongTensor(np.array(data))
        batch_data["label"] = table.float()
        # batch_data["sample_id"] = torch.LongTensor(sample_id)
        return batch_data

    def sampling_keys(self):
        return ["sample", "label"]
    
class CrossESampler(BaseSampler):

    def __init__(self, args):
        super().__init__(args)
        self.neg_weight = float(self.args.neg_weight / self.args.num_ent)
    def sampling(self, data):
        '''一个样本同时做head/tail prediction'''
        batch_data = {}
        hr_label = self.init_label(len(data))
        tr_label = self.init_label(len(data))
        for id, (h, r, t) in enumerate(data):
            hr_sample = self.hr2t_train[(h, r)]
            hr_label[id][hr_sample] = 1.0
            tr_sample = self.rt2h_train[(r, t)]
            tr_label[id][tr_sample] = 1.0
        batch_data["sample"] = torch.LongTensor(data)
        batch_data["hr_label"] = hr_label.float()
        batch_data["tr_label"] = tr_label.float()
        return batch_data

    def init_label(self, row):
        label = torch.rand(row, self.args.num_ent)
        label = (label > self.neg_weight).float()
        label -= 1.0
        return label

    def sampling_keys(self):
        return ["sample", "label"]

class ConvSampler(RevSampler): #TODO:SEGNN
    """Merging triples which have same head and relation, all false tail entities are taken as negative samples.      
    
    The triples which have same head and relation are treated as one triple.

    Attributes:
        label: Mask the false tail as negative samples.
        triples: The triples used to be sampled.
    """
    def __init__(self, args):
        self.label = None
        self.triples = None
        super().__init__(args)
        super().get_hr_trian()

    def sampling(self, pos_hr_t):
        """Randomly sampling from the merged triples.
    
        Args:
            pos_hr_t: The triples ((head,relation) pairs) used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        t_triples = []
        self.label = torch.zeros(self.args.train_bs, self.args.num_ent)
        self.triples  = torch.LongTensor([hr for hr , _ in pos_hr_t])
        for hr, t in pos_hr_t:
            t_triples.append(t)
        
        for id, hr_sample in enumerate([t for _ ,t in pos_hr_t]):
            self.label[id][hr_sample] = 1
    
        batch_data["sample"] = self.triples
        batch_data["label"] = self.label
        batch_data["t_triples"] = t_triples
        
        return batch_data

    def sampling_keys(self):
        return ["sample", "label", "t_triples"]
    
class XTransESampler(RevSampler):
    """Random negative sampling and recording neighbor entities.

    Attributes:
        triples: The triples used to be sampled.
        neg_sample: The negative samples.
        h_neighbor: The neighbor of sampled entites.
        h_mask: The tag of effecitve neighbor.
        max_neighbor: The maximum of the neighbor entities.
    """

    def __init__(self, args):
        super().__init__(args)
        super().get_h2rt_t2hr_from_train()
        self.triples    = None
        self.neg_sample = None
        self.h_neighbor = None
        self.h_mask     = None
        self.max_neighbor = 200

    def sampling(self, data):
        """Random negative sampling and recording neighbor entities.
    
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        
        neg_ent_sample = []
        mask = np.zeros([self.args.train_bs, 20000], dtype=float)
        h_neighbor = np.zeros([self.args.train_bs, 20000, 2])
        
        for id, triples in enumerate(data):
            h,r,t = triples
            num_h_neighbor = len(self.h2rt_train[h]) 
            h_neighbor[id][0:num_h_neighbor] = np.array(self.h2rt_train[h])
            
            mask[id][0:num_h_neighbor] = np.ones([num_h_neighbor])
            
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_ent_sample.append(neg_tail)

        self.triples    = data
        self.neg_sample = neg_ent_sample
        self.h_neighbor = h_neighbor[:, :self.max_neighbor]
        self.h_mask     = mask[:, :self.max_neighbor]

        batch_data["positive_sample"] = torch.LongTensor(self.triples)
        batch_data['negative_sample'] = torch.LongTensor(self.neg_sample)
        batch_data['neighbor']        = torch.LongTensor(self.h_neighbor)
        batch_data['mask']            = torch.LongTensor(self.h_mask)
        batch_data['mode']            = "tail-batch"
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'neighbor', 'mask', 'mode']

class GraphSampler(RevSampler):
    """Graph based sampling in neural network.

    Attributes:
        entity: The entities of sampled triples. 
        relation: The relation of sampled triples.
        triples: The sampled triples.
        graph: The graph structured sampled triples by dgl.graph in DGL.
        norm: The edge norm in graph.
        label: Mask the false tail as negative samples.
    """
    def __init__(self, args):
        super().__init__(args)
        self.entity   = None
        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None

    def sampling(self, pos_triples):
        """Graph based sampling in neural network.

        Args:
            pos_triples: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        
        pos_triples = np.array(pos_triples)
        pos_triples, self.entity = self.sampling_positive(pos_triples)
        head_triples = self.sampling_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sampling_negative('tail', pos_triples, self.args.num_neg)
        self.triples = np.concatenate((pos_triples,head_triples,tail_triples))
        batch_data['entity']  = self.entity
        batch_data['triples'] = self.triples
        
        self.label = torch.zeros((len(self.triples),1))
        self.label[0 : self.args.train_bs] = 1
        batch_data['label'] = self.label
        
        split_size = int(self.args.train_bs * 0.5) 
        graph_split_ids = np.random.choice(
            self.args.train_bs,
            size=split_size, 
            replace=False
        )
        head,rela,tail = pos_triples.transpose()
        head = torch.tensor(head[graph_split_ids], dtype=torch.long).contiguous()
        rela = torch.tensor(rela[graph_split_ids], dtype=torch.long).contiguous()
        tail = torch.tensor(tail[graph_split_ids], dtype=torch.long).contiguous()
        self.graph, self.relation, self.norm = self.build_graph(len(self.entity), (head,rela,tail), -1)
        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['graph','triples','label','entity','relation','norm']

    def sampling_negative(self, mode, pos_triples, num_neg):
        """Random negative sampling without filtering

        Args:
            mode: The mode of negtive sampling.
            pos_triples: The positive triples.
            num_neg: The number of negative samples corresponding to each triple.

        Results:
            neg_samples: The negative triples.
        """
        neg_random = np.random.choice(
            len(self.entity), 
            size = num_neg * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_samples[:,0] = neg_random
        elif mode == 'tail':
            neg_samples[:,2] = neg_random
        return neg_samples

    def build_graph(self, num_ent, triples, power):
        """Using sampled triples to build a graph by dgl.graph in DGL.

        Args:
            num_ent: The number of entities.
            triples: The positive sampled triples.
            power: The power index for normalization.

        Returns:
            rela: The relation of sampled triples.
            graph: The graph structured sampled triples by dgl.graph in DGL.
            edge_norm: The edge norm in graph.
        """
        head, rela, tail = triples[0], triples[1], triples[2]
        graph = dgl.graph(([], []))
        graph.add_nodes(num_ent)
        graph.add_edges(head, tail)
        node_norm = self.comp_deg_norm(graph, power)
        edge_norm = self.node_norm_to_edge_norm(graph,node_norm)
        rela = torch.tensor(rela)
        return graph, rela, edge_norm

    def comp_deg_norm(self, graph, power=-1):
        """Calculating the normalization node weight.

        Args:
            graph: The graph structured sampled triples by dgl.graph in DGL.
            power: The power index for normalization.

        Returns:
            tensor: The node weight of normalization.
        """
        graph = graph.local_var()
        in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        norm = in_deg.__pow__(power)
        norm[np.isinf(norm)] = 0
        return torch.from_numpy(norm)

    def node_norm_to_edge_norm(slef, graph, node_norm):
        """Calculating the normalization edge weight.

        Args:
            graph: The graph structured sampled triples by dgl.graph in DGL.
            node_norm: The node weight of normalization.

        Returns:
            tensor: The edge weight of normalization.
        """
        graph = graph.local_var()
        # convert to edge norm
        graph.ndata['norm'] = node_norm.view(-1,1)
        graph.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        return graph.edata['norm']

    def sampling_positive(self,positive_triples):
        """Regenerate positive sampling.

        Args:
            positive_triples: The positive sampled triples.

        Results:
            The regenerate triples and entities filter invisible entities.
        """

        edges = np.random.choice(
            np.arange(len(positive_triples)),
            size = self.args.train_bs,
            replace=False
        )
        edges = positive_triples[edges]
        head, rela, tail = np.array(edges).transpose()
        entity, index = np.unique((head, tail), return_inverse=True) 
        head, tail = np.reshape(index, (2, -1))

        return np.stack((head,rela,tail)).transpose(), \
                torch.from_numpy(entity).view(-1,1).long()

class KBATSampler(BaseSampler):
    """Graph based n_hop neighbours in neural network.

    Attributes:
        n_hop: The graph of n_hop neighbours.
        graph: The adjacency graph.
        neighbours: The neighbours of sampled triples.
        adj_matrix:The triples of sampled.
        triples: The sampled triples.
        triples_GAT_pos: Positive triples.
        triples_GAT_neg: Negative triples.
        triples_Con: All triples including positive triples and negative triples. 
        label: Mask the false tail as negative samples.
    """
    def __init__(self, args): 
        super().__init__(args)
        self.n_hop           = None
        self.graph           = None
        self.neighbours      = None
        self.adj_matrix      = None
        self.entity          = None
        self.triples_GAT_pos = None
        self.triples_GAT_neg = None
        self.triples_Con     = None
        self.label           = None

        self.get_neighbors()

    def sampling(self, pos_triples):
        """Graph based n_hop neighbours in neural network.

        Args:
            pos_triples: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        #--------------------KBAT-Sampler------------------------------------------
        self.entity = self.get_unique_entity(pos_triples)
        head_triples = self.sam_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sam_negative('tail', pos_triples, self.args.num_neg)
        self.triples_GAT_neg = torch.tensor(np.concatenate((head_triples, tail_triples)))
        batch_data['triples_GAT_pos'] = torch.tensor(pos_triples)
        batch_data['triples_GAT_neg'] = self.triples_GAT_neg

        head, rela, tail = torch.tensor(self.train_triples).t()
        self.adj_matrix  = (torch.stack((tail, head)), rela)
        batch_data['adj_matrix'] = self.adj_matrix

        self.n_hop = self.get_batch_nhop_neighbors_all()
        batch_data['n_hop'] = self.n_hop
        #--------------------ConvKB-Sampler------------------------------------------
        head_triples = self.sampling_negative('head', pos_triples, self.args.num_neg)
        tail_triples = self.sampling_negative('tail', pos_triples, self.args.num_neg)
        self.triples_Con = np.concatenate((pos_triples, head_triples, tail_triples))
        self.label = -torch.ones((len(self.triples_Con),1))
        self.label[0 : self.args.train_bs] = 1
        batch_data['triples_Con'] = self.triples_Con
        batch_data['label'] = self.label

        return batch_data

    def get_sampling_keys(self):
        return ['adj_matrix', 'n_hop', 'triples_GAT_pos', 
        'triples_GAT_neg', 'triples_Con' , 'label']

    def bfs(self, graph, source, nbd_size=2):
        """Using depth first search algorithm to generate n_hop neighbor graph.
        
        Args:
            graph: The adjacency graph.
            source: Head node.
            nbd_size: The number of hops.

        Returns:
            neighbors: N_hop neighbor graph.
        """
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target]) # 记录父亲节点id和关系id

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1]))) #删除已知的source 记录前两跳实体及关系
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_neighbors(self, nbd_size=2):
        """Getting the relation and entity of the source in the n_hop neighborhood.
        
        Args:
            nbd_size: The number of hops.

        Returns:
            self.neighbours: Record the relation and entity of the source in the n_hop neighborhood.
        """
        self.graph = {}

        for triple in self.train_triples:
            head = triple[0]
            rela = triple[1]
            tail = triple[2]

            if(head not in self.graph.keys()):
                self.graph[head] = {}
                self.graph[head][tail] = rela
            else:
                self.graph[head][tail] = rela

        neighbors = {}
        '''
        import pickle
        print("Opening node_neighbors pickle object")
        file = self.args.data_path + "/2hop.pickle"
        with open(file, 'rb') as handle:
            self.neighbours = pickle.load(handle)  
        return
        '''
        start_time = time.time()
        print("Start Graph BFS")
        for head in self.graph.keys():
            temp_neighbors = self.bfs(self.graph, head, nbd_size)
            for distance in temp_neighbors.keys():
                if(head in neighbors.keys()):
                    if(distance in neighbors[head].keys()):
                        neighbors[head][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[head][distance] = temp_neighbors[distance]
                else:
                    neighbors[head] = {}
                    neighbors[head][distance] = temp_neighbors[distance]

        print("Finish BFS, time taken ", time.time() - start_time)
        self.neighbours = neighbors

    def get_unique_entity(self, triples):
        """Getting the set of entity.
        
        Args:
            triples: The sampled triples.

        Returns:
            numpy.array: The set of entity
        """
        train_triples = np.array(triples)
        train_entities = np.concatenate((train_triples[:,0], train_triples[:,2]))
        return np.unique(train_entities)

    def get_batch_nhop_neighbors_all(self, nbd_size=2):
        """Getting n_hop neighbors of all entities in batch.
        
        Args:
            nbd_size: The number of hops.

        Returns:
            The set of n_hop neighbors.
        """
        batch_source_triples = []
        
        for source in self.entity:
            if source in self.neighbours.keys():
                nhop_list = self.neighbours[source][nbd_size]
                for i, tup in enumerate(nhop_list):
                    if(self.args.partial_2hop and i >= 2): 
                        break
                    batch_source_triples.append([source, 
                                                tup[0][-1], 
                                                tup[0][0],
                                                tup[1][0]])

        n_hop =  np.array(batch_source_triples).astype(np.int32)
        
        return torch.autograd.Variable(torch.LongTensor(n_hop))

    def sampling_negative(self, mode, pos_triples, num_neg):
        """Random negative sampling.

        Args:
            mode: The mode of negtive sampling.
            pos_triples: The positive triples.
            num_neg: The number of negative samples corresponding to each triple.

        Results:
            neg_samples: The negative triples.
        """
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_head = []
            for h, r, t in pos_triples:
                neg_head.append(self.head_batch(h, r, t, num_neg))
            neg_samples[:,0] = torch.tensor(neg_head).t().reshape(-1)
        elif mode == 'tail':
            neg_tail = []
            for h, r, t in pos_triples:
                neg_tail.append(self.tail_batch(h, r, t, num_neg))
            neg_samples[:,2] = torch.tensor(neg_tail).t().reshape(-1)
        return neg_samples

    def sam_negative(self, mode, pos_triples, num_neg):
        """Random negative sampling without filter.

        Args:
            mode: The mode of negtive sampling.
            pos_triples: The positive triples.
            num_neg: The number of negative samples corresponding to each triple.

        Results:
            neg_samples: The negative triples.
        """ 
        neg_random = np.random.choice(
            len(self.entity), 
            size = num_neg * len(pos_triples)
        )
        neg_samples = np.tile(pos_triples, (num_neg, 1))
        if mode == 'head':
            neg_samples[:,0] = neg_random
        elif mode == 'tail':
            neg_samples[:,2] = neg_random
        return neg_samples

class CompGCNSampler(GraphSampler):
    """Graph based sampling in neural network.

    Attributes:
        relation: The relation of sampled triples.
        triples: The sampled triples.
        graph: The graph structured sampled triples by dgl.graph in DGL.
        norm: The edge norm in graph.
        label: Mask the false tail as negative samples.
    """
    def __init__(self, args):
        super().__init__(args)
        self.relation = None
        self.triples  = None
        self.graph    = None
        self.norm     = None
        self.label    = None
        
        super().get_hr_trian()
        
        self.graph, self.relation, self.norm = \
            self.build_graph(self.args.num_ent, np.array(self.t_triples).transpose(), -0.5)

    def sampling(self, pos_hr_t):
        """Graph based n_hop neighbours in neural network.

        Args:
            pos_hr_t: The triples(hr, t) used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        
        self.label = torch.zeros(self.args.train_bs, self.args.num_ent)
        self.triples  = torch.LongTensor([hr for hr , _ in pos_hr_t])
        for id, hr_sample in enumerate([t for _ ,t in pos_hr_t]):
            self.label[id][hr_sample] = 1

        batch_data['sample']   = self.triples
        batch_data['label']    = self.label
        batch_data['graph']    = self.graph
        batch_data['relation'] = self.relation
        batch_data['norm']     = self.norm

        return batch_data

    def get_sampling_keys(self):
        return ['sample','label','graph','relation','norm']

    def node_norm_to_edge_norm(self, graph, node_norm):
        """Calculating the normalization edge weight.

        Args:
            graph: The graph structured sampled triples by dgl.graph in DGL.
            node_norm: The node weight of normalization.

        Returns:
            norm: The edge weight of normalization.
        """
        graph.ndata['norm'] = node_norm
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        norm = graph.edata.pop('norm').squeeze()
        return norm

class TestSampler(object):
    """Sampling triples and recording positive triples for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label"]

class ValidSampler(object):
    """Sampling subgraphs for validation.

    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args

    def sampling(self, data):
        """Sampling function to collect batch of subgraph for validation.

        Args:
            data: List of subgraph data for validation.

        Returns:
            batch_data: The batch of validating data.
        """
        batch_data = {}

        graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))

        graphs_neg = [item for sublist in graphs_negs for item in sublist]
        g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = (graphs_pos, r_labels_pos)
        batch_data["negative_sample"] = (graphs_neg, r_labels_neg)
        batch_data["graph_pos_label"] =  g_labels_pos
        batch_data["graph_neg_label"] =  g_labels_neg

        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'graph_label_pos', 'graph_label_neg']

class ValidRMPISampler(object): 
    """Sampling subgraphs for RMPI validation.
    
    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args

    def sampling(self, data):
        """Sampling function to collect batch of RMPI subgraph(enclosing and disclosing) for validation.

        Args:
            data: List of subgraph data for RMPI validation.

        Returns:
            batch_data: The batch of RMPI validating data.
        """
        batch_data = {}

        en_graphs_pos, dis_graphs_pos, g_labels_pos, r_labels_pos, en_graphs_negs, dis_graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))
        batched_en_graph_pos = dgl.batch(en_graphs_pos)
        batched_dis_graph_pos = dgl.batch(dis_graphs_pos)

        en_graphs_neg = [item for sublist in en_graphs_negs for item in sublist]
        dis_graphs_neg = [item for sublist in dis_graphs_negs for item in sublist]
        g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
        
        batched_en_graph_neg = dgl.batch(en_graphs_neg)
        batched_dis_graph_neg = dgl.batch(dis_graphs_neg)

        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = ((batched_en_graph_pos, batched_dis_graph_pos), r_labels_pos)
        batch_data["negative_sample"] = ((batched_en_graph_neg, batched_dis_graph_neg), r_labels_neg)
        batch_data["graph_pos_label"] =  g_labels_pos
        batch_data["graph_neg_label"] =  g_labels_neg

        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'graph_label_pos', 'graph_label_neg']

class TestSampler_hit(object):
    """Sampling subgraphs for testing link prediction.
    
    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
        m_h2r: The matrix of head to rels.
        m_t2r: The matrix of tail to rels.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args
        self.m_h2r = sampler.m_h2r
        self.m_t2r = sampler.m_t2r

    def sampling(self, data): # NOTE: data or test  固定test_bs为1 每次只取一个
        """Sampling function to collect batch of subgraph for testing mrr and hit@1,5,10.

        Args:
            data: List of subgraph data for testing.

        Returns:
            batch_data: The batch of testing data.
        """
        batch_data = {}

        test = data[0]
        head_neg_links = test['head'][0]
        tail_neg_links = test['tail'][0]

        batch_data['head_sample'] = self.get_subgraphs(head_neg_links, self.sampler.adj_list, \
                self.sampler.dgl_adj_list, self.args.max_n_label, self.m_h2r, self.m_t2r)
        batch_data['tail_sample'] = self.get_subgraphs(tail_neg_links, self.sampler.adj_list, \
                self.sampler.dgl_adj_list, self.args.max_n_label, self.m_h2r, self.m_t2r)
        batch_data['head_target'] = test['head'][1]
        batch_data['tail_target'] = test['tail'][1]
        return batch_data

    def get_sampling_keys(self):
        return ['head_sample', 'tail_sample', 'head_target', 'tail_target']

    def get_subgraphs(self, all_links, adj_list, dgl_adj_list, max_node_label_value, m_h2r, m_t2r):
        """Extracting and labeling subgraphs.

        Args:
            all_links: All head or tail entities link to corresponding triple.
            adj_list: List of adjacency matrix.
            dgl_adj_list: List of undirected head to tail matrix.
            max_node_label_value: Max value of node label.
            m_h2r: The matrix of head to rels.
            m_t2r: The matrix of tail to rels.

        Returns:
            subgraphs: Subgraphs for testing.
            r_labels: Labels of relation.
        """
        subgraphs = []
        r_labels = []

        for link in all_links:
            head, tail, rel = link[0], link[1], link[2]
            nodes, node_labels, _, _, _, _, _ = subgraph_extraction_labeling((head, tail), rel, adj_list, h=self.args.hop,
                            enclosing_sub_graph=self.args.enclosing_sub_graph, max_node_label_value=max_node_label_value)

            subgraph = dgl_adj_list.subgraph(nodes)
            subgraph.edata['type'] = dgl_adj_list.edata['type'][subgraph.edata[dgl.EID]]
            subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

            try:
                edges_btw_roots = subgraph.edge_ids(torch.LongTensor([0]), torch.LongTensor([1]))
            except:
                edges_btw_roots = torch.LongTensor([])

            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)
            if rel_link.squeeze().nelement() == 0:
                subgraph = dgl.add_edges(subgraph, torch.tensor([0]), torch.tensor([1]),
                                        {'type': torch.LongTensor([rel]),
                                        'label': torch.LongTensor([rel])})

            if len(m_h2r) != 0 and len(m_t2r) != 0:
                subgraph.ndata['out_nei_rels'] = torch.LongTensor(m_h2r[subgraph.ndata[dgl.NID]])
                subgraph.ndata['in_nei_rels'] = torch.LongTensor(m_t2r[subgraph.ndata[dgl.NID]])
                subgraph.ndata['r_label'] = torch.LongTensor(np.ones(subgraph.number_of_nodes()) * rel)

            subgraph = self.prepare_features(subgraph, node_labels, max_node_label_value)

            subgraphs.append(subgraph)
            r_labels.append(rel)
            
        r_labels = torch.LongTensor(r_labels)

        return (subgraphs, r_labels)

    def prepare_features(self, subgraph, n_labels, max_n_label, n_feats=None):
        """One hot encode the node label feature and concat to n_featsure.

        Args:
            subgraph: Subgraph for processing.
            n_labels: Node labels.
            max_n_label: Max value of node label.
            n_feats: node features.

        Returns:
            subgraph: Subgraph after processing.
        """
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph

class TestRMPISampler_hit(object): 
    """Sampling subgraphs for RMPI testing link prediction.
    
    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args

    def sampling(self, data): # NOTE: data or test 
        """Sampling function to collect batch of subgraph for RMPI testing.

        Args:
            data: List of subgraph data for RMPI sting.

        Returns:
            batch_data: The batch of RMPI testing data.
        """
        batch_data = {}

        test = data[0]
        head_neg_links = test['head'][0]
        tail_neg_links = test['tail'][0]

        batch_data['head_sample'] = self.get_subgraphs(head_neg_links, self.sampler.adj_list, \
                self.sampler.dgl_adj_list, self.args.max_n_label)
        batch_data['tail_sample'] = self.get_subgraphs(tail_neg_links, self.sampler.adj_list, \
            self.sampler.dgl_adj_list, self.args.max_n_label)
        batch_data['head_target'] = test['head'][1]
        batch_data['tail_target'] = test['tail'][1]
        return batch_data

    def get_sampling_keys(self):
        return ['head_sample', 'tail_sample', 'head_target', 'tail_target']

    def prepare_subgraph(self, dgl_adj_list, nodes, rel, node_labels, max_node_label_value):
        """Prepare enclosing or disclosing subgraph. 

        Args:
            dgl_adj_list: List of undirected head to tail matrix.
            nodes: Nodes of subgraph.
            rel: Relation idx.
            node_labels: Node labels.
            max_node_label_value: Max value of node label. 

        Returns:
            subgraph: Subgraph for testing.
        """
        subgraph = dgl_adj_list.subgraph(nodes)
        subgraph.edata['type'] = dgl_adj_list.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        try:
            edges_btw_roots = subgraph.edge_ids(torch.LongTensor([0]), torch.LongTensor([1]))
        except:
            edges_btw_roots = torch.LongTensor([])
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            subgraph = dgl.add_edges(subgraph, torch.tensor([0]), torch.tensor([1]),
                                     {'type': torch.LongTensor([rel]),
                                      'label': torch.LongTensor([rel])})
            e_ids = np.zeros(subgraph.number_of_edges())
            e_ids[-1] = 1  # target edge
        else:
            e_ids = np.zeros(subgraph.number_of_edges())
            e_ids[edges_btw_roots] = 1  # target edge

        subgraph.edata['id'] = torch.FloatTensor(e_ids)

        subgraph = self.prepare_features(subgraph, node_labels, max_node_label_value)
        return subgraph

    def get_subgraphs(self, all_links, adj_list, dgl_adj_list, max_node_label_value):
        """Extracting and labeling subgraphs.

        Args:
            all_links: All head or tail entities link to corresponding triple.
            adj_list: List of adjacency matrix.
            dgl_adj_list: List of undirected head to tail matrix.
            max_node_label_value: Max value of node label.

        Returns:
            subgraphs: Subgraphs for testing.
            r_labels: Labels of relation.
        """
        en_subgraphs = []
        dis_subgraphs = []
        r_labels = []

        for link in all_links:
            head, tail, rel = link[0], link[1], link[2]
            en_nodes, en_node_labels, _, _, _, dis_nodes, dis_node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=self.args.hop,
                                        enclosing_sub_graph=self.args.enclosing_sub_graph, max_node_label_value=max_node_label_value)

            en_subgraph = self.prepare_subgraph(dgl_adj_list, en_nodes, rel, en_node_labels, max_node_label_value)
            dis_subgraph = self.prepare_subgraph(dgl_adj_list, dis_nodes, rel, dis_node_labels, max_node_label_value)


            en_subgraphs.append(en_subgraph)
            dis_subgraphs.append(dis_subgraph)
            r_labels.append(rel)

        batched_en_graph = dgl.batch(en_subgraphs)
        batched_dis_graph = dgl.batch(dis_subgraphs)
        r_labels = torch.LongTensor(r_labels)

        return ((batched_en_graph, batched_dis_graph), r_labels)

    def prepare_features(self, subgraph, n_labels, max_n_label, n_feats=None):
        """One hot encode the node label feature and concat to n_featsure for RMPI.

        Args:
            subgraph: Subgraph for processing.
            n_labels: Node labels.
            max_n_label: Max value of node label.
            n_feats: node features.

        Returns:
            subgraph: Subgraph after processing.
        """
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph

class TestSampler_auc(object):
    """Sampling subgraphs for testing triple classification.
    
    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args

    def sampling(self, data):
        """Sampling function to collect batch of subgraph for testing auc and auc_pr.

        Args:
            data: List of subgraph data for testing.

        Returns:
            batch_data: The batch of testing data.
        """
        batch_data = {}

        graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))

        graphs_neg = [item for sublist in graphs_negs for item in sublist]
        g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = (graphs_pos, r_labels_pos)
        batch_data["negative_sample"] = (graphs_neg, r_labels_neg)
        batch_data["graph_pos_label"] =  g_labels_pos
        batch_data["graph_neg_label"] =  g_labels_neg

        return batch_data

class TestRMPISampler_auc(object):
    """Sampling subgraphs for testing RMPI triple classification.
    
    Attributes:
        sampler: The function of training sampler.
        args: Model configuration parameters.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args

    def sampling(self, data):
        """Sampling function to collect batch of subgraph for RMPI testing auc and auc_pr.

        Args:
            data: List of subgraph data for RMPI testing.

        Returns:
            batch_data: The batch of RMPI testing data.
        """
        batch_data = {}

        en_graphs_pos, dis_graphs_pos, g_labels_pos, r_labels_pos, en_graphs_negs, dis_graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*data))
        batched_en_graph_pos = dgl.batch(en_graphs_pos)
        batched_dis_graph_pos = dgl.batch(dis_graphs_pos)

        en_graphs_neg = [item for sublist in en_graphs_negs for item in sublist]
        dis_graphs_neg = [item for sublist in dis_graphs_negs for item in sublist]
        g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
        r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
        
        batched_en_graph_neg = dgl.batch(en_graphs_neg)
        batched_dis_graph_neg = dgl.batch(dis_graphs_neg)

        r_labels_pos = torch.LongTensor(r_labels_pos)
        r_labels_neg = torch.LongTensor(r_labels_neg)

        batch_data["positive_sample"] = ((batched_en_graph_pos, batched_dis_graph_pos), r_labels_pos)
        batch_data["negative_sample"] = ((batched_en_graph_neg, batched_dis_graph_neg), r_labels_neg)
        
        batch_data["graph_pos_label"] =  g_labels_pos
        batch_data["graph_neg_label"] =  g_labels_neg

        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'graph_label_pos', 'graph_label_neg']

class MetaSampler(BaseMeta):
    """Sampling meta task and collecting train data for training.

    """
    def __init__(self, args):
        super().__init__(args)
    
    def sampling(self, data):
        """Sampling function to collect batch of meta task for training, which is default.

        Args:
            data: List of task for training.

        Returns:
            data: List of task for training.
        """
        return data

    def get_sampling_keys(self):
        return []

class ValidMetaSampler(object):
    """Collecting task for validating.

    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args
    
    def sampling(self, data):
        """Sampling function to collect batch of meta task for validating, which is default.

        Args:
            data: List of task for validating.

        Returns:
            data: List of task for validating.
        """
        return data

    def get_sampling_keys(self):
        return []

class TestMetaSampler_hit(object):
    """Collecting task for testing.

    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args
    
    def sampling(self, data):
        """Sampling function to collect batch of meta task for testing mrr and hit@1,5,10.

        Args:
            data: List of task for testing.

        Returns:
            batch_data: Batch of task for testing mrr and hit@1,5,10.
        """
        batch_data = {}

        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_cand = torch.stack([_[1] for _ in data], dim=0)
        head_cand = torch.stack([_[2] for _ in data], dim=0)

        batch_data["positive_sample"] = pos_triple
        batch_data["tail_cand"] = tail_cand
        batch_data["head_cand"] =  head_cand
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "tail_cand", "head_cand"]

class TestMetaSampler_auc(object):
    """Collecting task for testing.

    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.args = sampler.args
    
    def sampling(self, data):
        """Sampling function to collect batch of meta task for testing auc and auc_pr.

        Args:
            data: List of task for testing.

        Returns:
            batch_data: Batch of task for testing auc and auc_pr.
        """
        batch_data = {}

        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_cand = torch.stack([_[1] for _ in data], dim=0)
        head_cand = torch.stack([_[2] for _ in data], dim=0)

        batch_data["positive_sample"] = pos_triple
        batch_data["positive_label"] = [1 for _ in pos_triple]
        neg_triples = []
        for idx, triple in enumerate(pos_triple):
            if np.random.uniform() < 0.5:
                neg_triples.append(torch.tensor([head_cand[idx][0].item(), triple[1].item(), triple[2].item()]))
            else:
                neg_triples.append(torch.tensor([triple[0].item(), triple[1].item(), tail_cand[idx][0].item()]))
        batch_data["negative_sample"] = torch.stack([_ for _ in neg_triples], dim=0)
        batch_data["negative_label"] = [0 for _ in pos_triple]
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'positive_label', 'negative_label']

class GraphTestSampler(object):
    """Sampling graph for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
        triples: The training triples.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.train_triples

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling graph for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        head, rela, tail = np.array(self.triples).transpose()
        graph, rela, norm = self.sampler.build_graph(self.num_ent, (head, rela, tail), -1)
        batch_data["graph"]  = graph
        batch_data["rela"]   = rela
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1,1)
        
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label",\
             "graph", "rela", "norm", "entity"]

class CompGCNTestSampler(object):
    """Sampling graph for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
        triples: The training triples.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent
        self.triples = sampler.t_triples

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling graph for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        
        for idx, triple in enumerate(data):
            # from IPython import embed;embed();exit()
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        
        graph, relation, norm = \
            self.sampler.build_graph(self.num_ent, np.array(self.triples).transpose(), -0.5)
    
        batch_data["graph"]  = graph
        batch_data["rela"]   = relation
        batch_data["norm"]   = norm
        batch_data["entity"] = torch.arange(0, self.num_ent, dtype=torch.long).view(-1,1)
        
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label",\
             "graph", "rela", "norm", "entity"]

class SEGNNTrainProcess(RevSampler):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.use_weight = self.args.use_weight
        #Parameters when constructing graph
        self.src_list = []
        self.dst_list = []
        self.rel_list = []
        self.hr2eid = ddict(list)
        self.rt2eid = ddict(list)

        self.ent_head = []
        self.ent_tail = []
        self.rel = []

        self.query = []
        self.label = []
        self.rm_edges = []
        self.set_scaling_weight = []

        self.hr2t_train_1 = ddict(set)
        self.ht2r_train_1 = ddict(set)
        self.rt2h_train_1 = ddict(set)
        self.get_h2rt_t2hr_from_train()
        self.construct_kg()
        self.get_sampling()
    
    def get_h2rt_t2hr_from_train(self):
        for h, r, t in self.train_triples:
            if r <= self.args.num_rel:
                self.ent_head.append(h)
                self.rel.append(r)
                self.ent_tail.append(t)
                self.hr2t_train_1[(h, r)].add(t)
                self.rt2h_train_1[(r, t)].add(h)
        
        for h, r in self.hr2t_train:
            self.hr2t_train_1[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train_1[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        h, r, t = self.query[item]
        label = self.get_onehot_label(self.label[item])

        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.args.rm_rate)
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num]
        rm_edges = rm_edges[rm_inds]

        return (h, r, t), label, rm_edges

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.args.num_ent)
        onehot_label[label] = 1
        if self.args.label_smooth != 0.0:
            onehot_label = (1.0 - self.args.label_smooth) * onehot_label + (1.0 / self.args.num_ent)

        return onehot_label
    

    def get_sampling(self):
        for k, v in self.hr2t_train_1.items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            self.rm_edges.append(self.hr2eid[k])
        
        for k, v in self.rt2h_train_1.items():
            self.query.append((k[1], k[0] + self.args.num_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(self.rt2eid[k])

    def construct_kg(self, directed=False):
        """
        construct kg.
        :param directed: whether add inverse version for each edge, to make a undirected graph.
        False when training SE-GNN model, True for comuting SE metrics.
        :return:
        """

        # eid: record the edge id of queries, for randomly removing some edges when training
        eid = 0
        for h, t, r in zip(self.ent_head, self.ent_tail, self.rel):
            if directed:
                self.src_list.extend([h])
                self.dst_list.extend([t])
                self.rel_list.extend([r])
                self.hr2eid[(h, r)].extend([eid])
                self.rt2eid[(r, t)].extend([eid])
                eid += 1
            else:
                # include the inverse edges
                # inverse rel id: original id + rel num
                self.src_list.extend([h, t])
                self.dst_list.extend([t, h])
                self.rel_list.extend([r, r + self.args.num_rel])
                self.hr2eid[(h, r)].extend([eid, eid + 1])
                self.rt2eid[(r, t)].extend([eid, eid + 1])
                eid += 2

        self.src_list, self.dst_list,self.rel_list = torch.tensor(self.src_list), torch.tensor(self.dst_list), torch.tensor(self.rel_list)

class SEGNNTrainSampler(object):
    def __init__(self, args):
        self.args = args
        self.get_train_1 = SEGNNTrainProcess(args)
        self.get_valid_1 = SEGNNTrainProcess(args).get_valid()
        self.get_test_1 = SEGNNTrainProcess(args).get_test()
    
    def get_train(self):
        return self.get_train_1
    def get_valid(self):
        return self.get_valid_1
    def get_test(self):
        return self.get_test_1
    
    def sampling(self, data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)  
        label = torch.stack(label, dim=0)  
        rm_edges = torch.cat(rm_edges, dim=0)  

        return (src, rel, dst), label, rm_edges

class SEGNNTestSampler(Dataset):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        #Parameters when constructing graph
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        for h, r, t in self.sampler.get_train_1.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            # self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        # for r, t in self.rt2h_all:
        #     self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.sampler.args.num_ent)
        tail_label = torch.zeros(len(data), self.sampler.args.num_ent)
        filter_head = torch.zeros(len(data), self.sampler.args.num_ent)
        filter_tail = torch.zeros(len(data), self.sampler.args.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            filter_tail[idx][self.hr2t_all[(head, rel)]] = -float('inf')
            filter_tail[idx][tail] = 0
            
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
       
        batch_data["filter_tail"] = filter_tail
       
        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "filter_tail", "tail_label"]
        
'''继承torch.Dataset'''
class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

