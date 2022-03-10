from numpy.random.mtrand import normal
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import random
from .DataPreprocess import *
from IPython import embed
import dgl 
import torch.nn.functional as F
import time
import queue

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
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.args.bern_flag else 0.5
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_head(t, r, num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_tail(h, r, num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return np.hstack((neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]))

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
            neg_ent_sample = self.__normal_batch(h, r, t, self.args.num_neg)
        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
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
    # TODO:类名还需要商榷下
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

class ConvSampler(RevSampler):
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

        self.label = torch.zeros(self.args.train_bs, self.args.num_ent)
        self.triples  = torch.LongTensor([hr for hr , _ in pos_hr_t])
        for id, hr_sample in enumerate([t for _ ,t in pos_hr_t]):
            self.label[id][hr_sample] = 1
    
        batch_data["sample"] = self.triples
        batch_data["label"] = self.label
        
        return batch_data

    def sampling_keys(self):
        return ["sample", "label"]

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
                    if(i >= 2): 
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

'''继承torch.Dataset'''
class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]