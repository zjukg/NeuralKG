import os
import dgl
import lmdb
import torch
import struct
import pickle
import logging
import numpy as np
from scipy.sparse import csc_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict as ddict
from neuralkg_ind.utils import deserialize, deserialize_RMPI, ssp_multigraph_to_dgl, gen_subgraph_datasets, get_indtest_test_dataset_and_train_g

class KGData(object):
    """Data preprocessing of kg data.

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc. 
        ent2id: Encoding the entity in triples, type: dict.
        rel2id: Encoding the relation in triples, type: dict.
        id2ent: Decoding the entity in triples, type: dict.
        id2rel: Decoding the realtion in triples, type: dict.
        train_triples: Record the triples for training, type: list.
        valid_triples: Record the triples for validation, type: list.
        test_triples: Record the triples for testing, type: list.
        all_true_triples: Record all triples including train,valid and test, type: list.
        TrainTriples
        Relation2Tuple
        RelSub2Obj
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
        h2rt_train: Record the tail, relation corresponding to the same head, type: defaultdict(class:set).
        t2rh_train: Record the head, realtion corresponding to the same tail, type: defaultdict(class:set).
    """

    def __init__(self, args):
        self.args = args

        #  基础部分
        self.ent2id = {}
        self.rel2id = {}
        # predictor需要
        self.id2ent = {}
        self.id2rel = {}
        # 存放三元组的id
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.all_true_triples = set()
        #  grounding 使用
        self.TrainTriples = {}
        self.Relation2Tuple = {}
        self.RelSub2Obj = {}

        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.get_id()
        self.get_triples_id()
        if args.use_weight:
            self.count = self.count_frequency(self.train_triples)

    def get_id(self):
        """Getting entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        with open(os.path.join(self.args.data_path, "entities.dict")) as fin:
            for line in fin:
                eid, entity = line.strip().split("\t")
                self.ent2id[entity] = int(eid)
                self.id2ent[int(eid)] = entity

        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation] = int(rid)
                self.id2rel[int(rid)] = relation

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

    def get_triples_id(self):
        """Getting triples id, save in the format of (h, r, t).

        Update:
            self.train_triples: Train dataset triples id.
            self.valid_triples: Valid dataset triples id.
            self.test_triples: Test dataset triples id.
        """
        
        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )
                
                tmp = str(self.ent2id[h]) + '\t' + str(self.rel2id[r]) + '\t' + str(self.ent2id[t])
                self.TrainTriples[tmp] = True

                iRelationID = self.rel2id[r]
                strValue = str(h) + "#" + str(t)
                if not iRelationID in self.Relation2Tuple:
                    tmpLst = []
                    tmpLst.append(strValue)
                    self.Relation2Tuple[iRelationID] = tmpLst
                else:
                    self.Relation2Tuple[iRelationID].append(strValue)

                iRelationID = self.rel2id[r]
                iSubjectID = self.ent2id[h]
                iObjectID = self.ent2id[t]
                tmpMap = {}
                tmpMap_in = {}
                if not iRelationID in self.RelSub2Obj:
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap
                else:
                    tmpMap = self.RelSub2Obj[iRelationID]
                    if not iSubjectID in tmpMap:
                        tmpMap_in.clear()
                        tmpMap_in[iObjectID] = True
                        tmpMap[iSubjectID] = tmpMap_in
                    else:
                        tmpMap[iSubjectID][iObjectID] = True
                    self.RelSub2Obj[iRelationID] = tmpMap  # 是不是应该要加？

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_hr2t_rt2h_from_train(self):
        """Getting the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """
        
        for h, r, t in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    @staticmethod
    def count_frequency(triples, start=4):
        '''Getting frequency of a partial triple like (head, relation) or (relation, tail).
        
        The frequency will be used for subsampling like word2vec.
        
        Args:
            triples: Sampled triples.
            start: Initial count number.

        Returns:
            count: Record the number of (head, relation).
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
        

    def get_h2rt_t2hr_from_train(self):
        """Getting the set of h2rt and t2hr from train dataset, the data type is numpy.

        Update:
            self.h2rt_train: The set of h2rt.
            self.t2rh_train: The set of t2hr.
        """
        for h, r, t in self.train_triples:
            self.h2rt_train[h].add((r, t))
            self.t2rh_train[t].add((r, h))
        for h in self.h2rt_train:
            self.h2rt_train[h] = np.array(list(self.h2rt_train[h]))
        for t in self.t2rh_train:
            self.t2rh_train[t] = np.array(list(self.t2rh_train[t]))
        
    def get_hr_trian(self):
        '''Change the generation mode of batch.
        Merging triples which have same head and relation for 1vsN training mode.

        Returns:
            self.train_triples: The tuple(hr, t) list for training
        '''
        self.t_triples = self.train_triples 
        self.train_triples = [ (hr, list(t)) for (hr,t) in self.hr2t_train.items()]

class GRData(Dataset):
    """Data preprocessing of subgraph. -- DGL Only

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc. 
        db_name_pos: Database name of positive sample, type: str.
        db_name_neg: Database name of negative sample, type: str.
        m_h2r: The matrix of head to rels, type: NDArray[signedinteger].
        m_t2r: The matrix of tail to rels, type: NDArray[signedinteger].
        ssp_graph: The collect of head to tail csc_matrix. type: list. 
        graph: Dgl graph of train or test, type: DGLHeteroGraph.
        id2entity: Record the id to entity. type: dict.
        id2relation: Record the id to relation. type: dict.
    """

    def __init__(self, args, db_name_pos, db_name_neg):
        
        self.args = args
        self.max_dbs = 5

        self.m_h2r = None
        self.m_t2r = None
        
        if db_name_pos == 'test_pos':
            self.main_env = lmdb.open(self.args.test_db_path, readonly=True, max_dbs=self.max_dbs, lock=False)
        else:
            self.main_env = lmdb.open(self.args.db_path, readonly=True, max_dbs=self.max_dbs, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())

        if db_name_pos == 'test_pos':
            ssp_graph, __, __, relation2id, id2entity, id2relation, _, m_h2r, _, m_t2r = self.load_ind_data_grail()
        else:
            ssp_graph, __, __, relation2id, id2entity, id2relation, _, m_h2r, _, m_t2r = self.load_data_grail()
        self.relation2id = relation2id
        
        if db_name_pos == 'train_pos':
            self.args.num_rel = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if self.args.add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        if db_name_pos == 'train_pos':
            self.args.aug_num_rels = len(ssp_graph)

        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        
        if self.args.model_name == 'SNRI':
            self.m_h2r = m_h2r
            self.m_t2r = m_t2r

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')
            if self.args.model_name != 'RMPI':
                self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
                self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
                self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
                self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

                self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
                self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
                self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
                self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

                self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
                self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
                self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
                self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        
        self.args.max_n_label = self.max_n_label

        self.__getitem__(0)

    def __getitem__(self, index):
        '''Getting the subgraph corresponding to the index and prepare subgraph features. 
        
        Args:
            index: Index of triple, which can obtain corresponding subgraph nodes from lmdb.

        Returns:
            subgraph_pos: Enclosing subgraph corresponding to positive sample.
            dis_subgraph_pos: Disclosing subgraph corresponding to positive sample.
            g_label_pos: The label of positive sample subgraph.
            r_labels_pos: The label of positive sample triple relation. 
            subgraphs_neg: Enclosing subgraph corresponding to negative sample.
            dis_subgraph_neg: Disclosing subgraph corresponding to negative sample.
            g_label_neg: The label of negative sample subgraph.
            r_labels_neg: The label of negative sample triple relation. 
        '''
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            if self.args.model_name == 'RMPI':
                en_nodes_pos, r_label_pos, g_label_pos, en_n_labels_pos, dis_nodes_pos, dis_n_labels_pos = deserialize_RMPI(txn.get(str_id)).values()
                subgraph_pos = self.prepare_subgraphs(en_nodes_pos, r_label_pos, en_n_labels_pos)
                dis_subgraph_pos = self.prepare_subgraphs(dis_nodes_pos, r_label_pos, dis_n_labels_pos)
            else:
                nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
                subgraph_pos = self.prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        dis_subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.args.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                if self.args.model_name == 'RMPI':
                    en_nodes_neg, r_label_neg, g_label_neg, en_n_labels_neg, dis_nodes_neg, dis_n_labels_neg = deserialize_RMPI(txn.get(str_id)).values()
                    subgraphs_neg.append(self.prepare_subgraphs(en_nodes_neg, r_label_neg, en_n_labels_neg))
                    dis_subgraphs_neg.append(self.prepare_subgraphs(dis_nodes_neg, r_label_neg, dis_n_labels_neg))
                else:
                    nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                    subgraphs_neg.append(self.prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        if self.args.model_name == 'RMPI':
            return subgraph_pos, dis_subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, dis_subgraphs_neg, g_labels_neg, r_labels_neg
        else:
            return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        '''Getting number of subgraph.

        Returns:
            num_graphs_pos: number of positive sample subgraph.
        '''
        return self.num_graphs_pos

    def load_data_grail(self):
        '''Load train dataset, adj_list, ent2idx, etc.

        Returns:
            adj_list: The collect of head to tail csc_matrix. type: list. 
            triplets: Triple of train-train and train-validation.
            train_ent2idx: Entity to idx of train graph.
            train_rel2idx: Relation to idx of train graph.
            train_idx2ent: idx to entity of train graph.
            train_idx2rel: idx to relation of train graph.
            h2r: Head to relation of train-train triple.
            m_h2r: The matrix of head to rels.
            t2r: Tail to relation of train-train triple.
            m_t2r: The matrix of tail to rels
        '''
        data = pickle.load(open(self.args.pk_path, 'rb'))

        splits = ['train', 'valid']
        
        triplets = {}
        for split_name in splits:
            triplets[split_name] = np.array(data['train_graph'][split_name])[:, [0, 2, 1]]

        train_rel2idx = data['train_graph']['rel2idx']
        train_ent2idx = data['train_graph']['ent2idx']
        train_idx2rel = {i: r for r, i in train_rel2idx.items()}
        train_idx2ent = {i: e for e, i in train_ent2idx.items()}

        h2r = {}
        t2r = {}
        m_h2r = {}
        m_t2r = {}
        if self.args.model_name == 'SNRI':
            # Construct the the neighbor relations of each entity
            num_rels = len(train_idx2rel)
            num_ents = len(train_idx2ent)
            h2r_len = {}
            t2r_len = {}
            
            for triplet in triplets['train']:
                h, t, r = triplet
                if h not in h2r:
                    h2r_len[h] = 1
                    h2r[h] = [r]
                else:
                    h2r_len[h] += 1
                    h2r[h].append(r)
                
                if self.args.add_traspose_rels:
                    # Consider the reverse relation, the id of reverse relation is (relation + #relations)
                    if t not in t2r:
                        t2r[t] = [r + num_rels]
                    else:
                        t2r[t].append(r + num_rels)
                if t not in t2r:
                    t2r[t] = [r]
                    t2r_len[t]  = 1
                else:
                    t2r[t].append(r)
                    t2r_len[t] += 1

            # Construct the matrix of ent2rels
            h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
            t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
            
            # The index "num_rels" of relation is considered as "padding" relation.
            # Use padding relation to initialize matrix of ent2rels.
            m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
            for ent, rels in h2r.items():
                if len(rels) > h_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
                    m_h2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_h2r[ent][: rels.shape[0]] = rels      
            
            m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
            for ent, rels in t2r.items():
                if len(rels) > t_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
                    m_t2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_t2r[ent][: rels.shape[0]] = rels

            # Sort the data according to relation id 
            if self.args.sort_data:
                triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]
        
        adj_list = []
        for i in range(len(train_rel2idx)):
            idx = np.argwhere(triplets['train'][:, 2] == i)
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                    shape=(len(train_ent2idx), len(train_ent2idx))))

        return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, h2r, m_h2r, t2r, m_t2r
    
    def load_ind_data_grail(self):
        '''Load test dataset, adj_list, ent2idx, etc.

        Returns:
            adj_list: The collect of head to tail csc_matrix. type: list. 
            triplets: Triple of test-train and test-test.
            train_ent2idx: Entity to idx of test graph.
            train_rel2idx: Relation to idx of test graph.
            train_idx2ent: idx to entity of test graph.
            train_idx2rel: idx to relation of test graph.
            h2r: Head to relation of test-train triple.
            m_h2r: The matrix of head to rels.
            t2r: Tail to relation of test-train triple.
            m_t2r: The matrix of tail to rels
        '''
        data = pickle.load(open(self.args.pk_path, 'rb'))

        splits = ['train', 'test']

        triplets = {}
        for split_name in splits:
            triplets[split_name] = np.array(data['ind_test_graph'][split_name])[:, [0, 2, 1]]

        train_rel2idx = data['ind_test_graph']['rel2idx']
        train_ent2idx = data['ind_test_graph']['ent2idx']
        train_idx2rel = {i: r for r, i in train_rel2idx.items()}
        train_idx2ent = {i: e for e, i in train_ent2idx.items()}

        h2r = {}
        t2r = {}
        m_h2r = {}
        m_t2r = {}
        if self.args.model_name == 'SNRI':
            # Construct the the neighbor relations of each entity
            num_rels = len(train_idx2rel)
            num_ents = len(train_idx2ent)
            h2r_len = {}
            t2r_len = {}
            
            for triplet in triplets['train']:
                h, t, r = triplet
                if h not in h2r:
                    h2r_len[h] = 1
                    h2r[h] = [r]
                else:
                    h2r_len[h] += 1
                    h2r[h].append(r)
                
                if self.args.add_traspose_rels:
                    # Consider the reverse relation, the id of reverse relation is (relation + #relations)
                    if t not in t2r:
                        t2r[t] = [r + num_rels]
                    else:
                        t2r[t].append(r + num_rels)
                if t not in t2r:
                    t2r[t] = [r]
                    t2r_len[t]  = 1
                else:
                    t2r[t].append(r)
                    t2r_len[t] += 1

            # Construct the matrix of ent2rels
            h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
            t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
            
            # The index "num_rels" of relation is considered as "padding" relation.
            # Use padding relation to initialize matrix of ent2rels.
            m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
            for ent, rels in h2r.items():
                if len(rels) > h_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
                    m_h2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_h2r[ent][: rels.shape[0]] = rels      
            
            m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
            for ent, rels in t2r.items():
                if len(rels) > t_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
                    m_t2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_t2r[ent][: rels.shape[0]] = rels

            # Sort the data according to relation id 
            if self.args.sort_data:
                triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]

        adj_list = []
        for i in range(len(train_rel2idx)):
            idx = np.argwhere(triplets['train'][:, 2] == i)
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                    shape=(len(train_ent2idx), len(train_ent2idx))))

        return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, h2r, m_h2r, t2r, m_t2r

    def generate_train(self):
        self.db_pos = self.main_env.open_db('train_pos'.encode())
        self.db_neg = self.main_env.open_db('train_neg'.encode())

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        return self.__getitem__(0)

    def generate_valid(self):
        self.db_pos = self.main_env.open_db('valid_pos'.encode())
        self.db_neg = self.main_env.open_db('valid_neg'.encode())

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        return self.__getitem__(0)

    def prepare_subgraphs(self, nodes, r_label, n_labels):
        '''Initialize subgraph nodes and relation characteristics.
        
        Args:
            nodes: The nodes of subgraph.
            r_label: The label of relation in subgraph corresponding triple.
            n_labels: The label of node in subgraph.

        Returns:
            subgraph: Subgraph after processing.
        '''
        subgraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        try:
            edges_btw_roots = subgraph.edge_ids(torch.LongTensor([0]), torch.LongTensor([1]))
        except:
            edges_btw_roots = torch.LongTensor([])
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph = dgl.add_edges(subgraph, torch.tensor([0]), torch.tensor([1]),
                                     {'type': torch.LongTensor([r_label]),
                                      'label': torch.LongTensor([r_label])})
            e_ids = np.zeros(subgraph.number_of_edges())
            e_ids[-1] = 1
        else:
            e_ids = np.zeros(subgraph.number_of_edges())
            e_ids[edges_btw_roots] = 1  # target edge
        
        if self.args.model_name == 'RMPI':
            subgraph.edata['id'] = torch.FloatTensor(e_ids)

        if  self.args.model_name == 'SNRI':
            subgraph = self.prepare_features_new(subgraph, n_labels, r_label)
        else:
            subgraph = self.prepare_features_new(subgraph, n_labels)
        if self.args.model_name == 'SNRI':
            # subgraph.ndata['parent_id'] = self.graph.subgraph(nodes).parent_nid
            subgraph.ndata['out_nei_rels'] = torch.LongTensor(self.m_h2r[subgraph.ndata[dgl.NID]])
            subgraph.ndata['in_nei_rels'] = torch.LongTensor(self.m_t2r[subgraph.ndata[dgl.NID]])

        return subgraph

    def prepare_features_new(self, subgraph, n_labels, r_label=None):
        '''prepare subgraph node features

        Args:
            subgraph: Extract subgraph.
            r_label: The label of relation in subgraph corresponding triple.
            n_labels: The label of node in subgraph.

        Returns:
            subgraph: Subgraph after initialize node label.
        
        '''
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1

        n_feats = label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        if self.args.model_name == 'SNRI':
            subgraph.ndata['r_label'] = torch.LongTensor(np.ones(n_nodes) * r_label)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

class MetaTrainGRData(Dataset):
    """Data preprocessing of meta train task.

    Attributes:
        subgraphs_db: database of train subgraphs.
    """
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=5, lock=False)
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())

    def __len__(self):
        return self.args.num_train_subgraph

    def __getitem__(self, idx):
        '''Getting the train meta task corresponding to the index. 
        
        Args:
            index: Index of train subgraph, which can obtain corresponding task triple from lmdb.

        Returns:
            sup_tri: Support triple in train task.
            que_tri: Query triple in train task.
            que_neg_tail_ent: Negative sample for query tail entity.
            que_neg_head_ent: Negative sample for query head entity.
        '''
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, que_tri, hr2t, rt2h = pickle.loads(txn.get(str_id))

        nentity = len(np.unique(np.array(sup_tri)[:, [0, 2]]))

        que_neg_tail_ent = [np.random.choice(np.delete(np.arange(nentity), hr2t[(h, r)]),
                                        self.args.num_neg) for h, r, t in que_tri]

        que_neg_head_ent = [np.random.choice(np.delete(np.arange(nentity), rt2h[(r, t)]),
                                        self.args.num_neg) for h, r, t in que_tri]

        return torch.tensor(sup_tri), torch.tensor(que_tri), \
               torch.tensor(np.array(que_neg_tail_ent)), torch.tensor(np.array(que_neg_head_ent))

class MetaValidGRData(Dataset):
    """Data preprocessing of meta valid task.

    Attributes:
        subgraphs_db: database of valid subgraphs.
    """
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=5, lock=False)
        self.subgraphs_db = self.env.open_db("valid_subgraphs".encode())

    def __len__(self):
        txn = self.env.begin(db=self.subgraphs_db)
        num = txn.stat()['entries']
        return num

    def __getitem__(self, idx):
        '''Getting the valid meta task corresponding to the index. 
        
        Args:
            index: Index of valid subgraph, which can obtain corresponding task triple from lmdb.

        Returns:
            sup_tri: Support triple in valid task.
            que_dataloader: Dataloader of query triple in valid task.
        '''
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, que_tri, hr2t, rt2h = pickle.loads(txn.get(str_id))

        nentity = len(np.unique(np.array(sup_tri)[:, [0, 2]]))

        que_dataset = KGEEvalData(self.args, que_tri, nentity, hr2t, rt2h)
        que_dataloader = DataLoader(que_dataset, batch_size=len(que_tri),
                                    collate_fn=KGEEvalData.collate_fn)

        return torch.tensor(sup_tri), que_dataloader

class KGEEvalData(Dataset):
    """Data processing for kge evaluate.

    Attributes: 
        triples: Evaluate triples. type: list.
        num_ent: The number of entity. type: int.
        hr2t: Head and raltion to tails. type: dict.
        rt2h: Relation and tail to heads. type: dict.
        num_cand: The number of candidate entities. type: str or int.
    """
    def __init__(self, args, eval_triples, num_ent, hr2t, rt2h):
        self.args = args
        self.triples = eval_triples
        self.num_ent = num_ent
        self.hr2t = hr2t
        self.rt2h = rt2h
        self.num_cand = 'all'

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        '''Sample negative candidate entities corresponding to the index. 
        
        Args:
            index: Index of triple.

        Returns:
            pos_triple: Positive triple.
            tail_cand: Candidate of tail entities.
            head_cand: Candidate of head entities.
        '''
        pos_triple = self.triples[idx]
        h, r, t = pos_triple
        if self.num_cand == 'all':
            tail_label, head_label = self.get_label(self.hr2t[(h, r)], self.rt2h[(r, t)])
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                             self.num_cand)

            neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                             self.num_cand)
            if self.args.eval_task == 'link_prediction':
                tail_cand = torch.from_numpy(np.concatenate(([t], neg_tail_cand)))
                head_cand = torch.from_numpy(np.concatenate(([h], neg_head_cand)))
            elif self.args.eval_task == 'triple_classification':
                tail_cand = torch.from_numpy(neg_tail_cand)
                head_cand = torch.from_numpy(neg_head_cand)
                
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_cand, head_cand

    def get_label(self, true_tail, true_head):
        '''Filter head and tail entities.

        Args:
            true_tail: Existing tail entities in dataset.
            true_head: Existing head entities in dataset.

        Returns:
            y_tail: Label of tail entities.
            y_head: Label of head entities.
        '''
        y_tail = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_tail:
            y_tail[e] = 1.0
        y_head = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_head:
            y_head[e] = 1.0

        return torch.FloatTensor(y_tail), torch.FloatTensor(y_head)

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_label_or_cand = torch.stack([_[1] for _ in data], dim=0)
        head_label_or_cand = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, tail_label_or_cand, head_label_or_cand

class BaseSampler(KGData):
    """Traditional random sampling mode.
    """
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()

    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples

class RevSampler(KGData):
    """Adding reverse triples in traditional random sampling mode.

    For each triple (h, r, t), generate the reverse triple (t, r`, h).
    r` = r + num_rel.

    Attributes:
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
    """
    def __init__(self, args):
        super().__init__(args)
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.add_reverse_relation()
        self.add_reverse_triples()
        self.get_hr2t_rt2h_from_train()

    def add_reverse_relation(self):
        """Getting entity/relation/reverse relation id, and entity/relation number.

        Update:
            self.ent2id: Entity id.
            self.rel2id: Relation id.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        
        with open(os.path.join(self.args.data_path, "relations.dict")) as fin:
            len_rel2id = len(self.rel2id)
            for line in fin:
                rid, relation = line.strip().split("\t")
                self.rel2id[relation + "_reverse"] = int(rid) + len_rel2id
                self.id2rel[int(rid) + len_rel2id] = relation + "_reverse"
        self.args.num_rel = len(self.rel2id)

    def add_reverse_triples(self):
        """Generate reverse triples (t, r`, h).

        Update:
            self.train_triples: Triples for training.
            self.valid_triples: Triples for validation.
            self.test_triples: Triples for testing.
            self.all_ture_triples: All triples including train, valid and test.
        """

        with open(os.path.join(self.args.data_path, "train.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.train_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "valid.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.valid_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        with open(os.path.join(self.args.data_path, "test.txt")) as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                self.test_triples.append(
                    (self.ent2id[t], self.rel2id[r + "_reverse"], self.ent2id[h])
                )

        self.all_true_triples = set(
            self.train_triples + self.valid_triples + self.test_triples
        )

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples    
    
    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated 

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

class BaseGraph(object):
    '''Base subgraph class

    collect train, valid, test dataset for inductive.
    '''
    def __init__(self, args):
        self.args = args
        self.train_triples = GRData(args, 'train_pos', 'train_neg')
        self.valid_triples = GRData(args, 'valid_pos', 'valid_neg')
        self.n_feat_dim = self.train_triples.n_feat_dim

        if self.args.model_name == 'SNRI':
            if args.init_nei_rels == 'no':
                args.inp_dim = self.n_feat_dim
            else:
                args.inp_dim = self.n_feat_dim + args.sem_dim
            self.valid_triples.m_h2r = self.train_triples.m_h2r
            self.valid_triples.m_t2r = self.train_triples.m_t2r

        if self.args.eval_task == 'link_prediction':
            self.test_triples = self.generate_ind_test()
            
        elif self.args.eval_task == 'triple_classification':
            if args.test_db_path is not None and not os.path.exists(args.test_db_path):
                gen_subgraph_datasets(args, splits=['test'],
                                    saved_relation2id=self.train_triples.relation2id,
                                    max_label_value=self.train_triples.max_n_label)

            self.test_triples = GRData(args, 'test_pos', 'test_neg')

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples
    
    def generate_ind_test(self):
        '''generate inductive test triples.

        Returns:
            neg_triplets: Negative triplets.
        '''
        adj_list, dgl_adj_list, triplets, m_h2r, m_t2r = self.load_data_grail_ind()
        neg_triplets = self.get_neg_samples_replacing_head_tail(triplets['test'], adj_list)
        
        self.adj_list = adj_list
        self.dgl_adj_list = dgl_adj_list
        self.m_h2r = m_h2r
        self.m_t2r = m_t2r

        return neg_triplets

    def load_data_grail_ind(self):
        '''Load train dataset, adj_list, ent2idx, etc.

        Returns:
            adj_list: The collect of head to tail csc_matrix.
            dgl_adj_list: The collect of undirected head to tail csc_matrix.
            triplets: Triple of test-train and test-test.
            m_h2r: The matrix of head to rels.
            m_t2r: The matrix of tail to rels
        '''
        data = pickle.load(open(self.args.pk_path, 'rb'))

        splits = ['train', 'test']

        triplets = {}
        for split_name in splits:
            triplets[split_name] = np.array(data['ind_test_graph'][split_name])[:, [0, 2, 1]]

        self.rel2id = data['ind_test_graph']['rel2idx']
        self.ent2id = data['ind_test_graph']['ent2idx']
        self.id2rel = {i: r for r, i in self.rel2id.items()}
        self.id2ent = {i: e for e, i in self.ent2id.items()}

        num_rels = len(self.id2rel)
        num_ents = len(self.id2ent)
        h2r = {}
        h2r_len = {}
        t2r = {}
        t2r_len = {}
        m_h2r = {}
        m_t2r = {}
        if self.args.model_name == 'SNRI':
            for triplet in triplets['train']:
                h, t, r = triplet
                if h not in h2r:
                    h2r_len[h] = 1
                    h2r[h] = [r]
                else:
                    h2r_len[h] += 1
                    h2r[h].append(r)
                
                if t not in t2r:
                    t2r[t] = [r]
                    t2r_len[t]  = 1
                else:
                    t2r[t].append(r)
                    t2r_len[t] += 1
                
            # Construct the matrix of ent2rels
            h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
            t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
            
            # The index "num_rels" of relation is considered as "padding" relation.
            # Use padding relation to initialize matrix of ent2rels.
            m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
            for ent, rels in h2r.items():
                if len(rels) > h_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
                    m_h2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_h2r[ent][: rels.shape[0]] = rels      
            
            m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
            for ent, rels in t2r.items():
                if len(rels) > t_nei_rels_len:
                    rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
                    m_t2r[ent] = rels
                else:
                    rels = np.array(rels)
                    m_t2r[ent][: rels.shape[0]] = rels   

        adj_list = []
        for i in range(len(self.rel2id)):
            idx = np.argwhere(triplets['train'][:, 2] == i)
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                    shape=(len(self.ent2id), len(self.ent2id))))

        adj_list_aug = adj_list
        if self.args.add_traspose_rels:
            adj_list_t = [adj.T for adj in adj_list]
            adj_list_aug = adj_list + adj_list_t
        
        dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

        return adj_list, dgl_adj_list, triplets, m_h2r, m_t2r

    def get_neg_samples_replacing_head_tail(self, test_links, adj_list, num_samples=50):
        '''Sample negative triplets by relacing head or tail.
        
        Args:
            test_links: test-test triplets.
            adj_list: The collect of head to tail csc_matrix.
            num_samples: The number of candidates.

        Returns:
            neg_triplets: Sampled negative triplets.
        '''
        n, r = adj_list[0].shape[0], len(adj_list)
        heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

        neg_triplets = []
        for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
            neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
            neg_triplet['head'][0].append([head, tail, rel])
            while len(neg_triplet['head'][0]) < num_samples:
                neg_head = head
                neg_tail = np.random.choice(n)

                if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                    neg_triplet['head'][0].append([neg_head, neg_tail, rel])

            neg_triplet['tail'][0].append([head, tail, rel])
            while len(neg_triplet['tail'][0]) < num_samples:
                neg_head = np.random.choice(n)
                neg_tail = tail
                # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

                if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                    neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

        return neg_triplets

class BaseMeta(object):
    """Base meta class

    collect train, valid, test dataset for meta task.
    """
    def __init__(self, args):
        self.args = args
        self.train_triples = MetaTrainGRData(args)
        self.valid_triples = MetaValidGRData(args)
        data, num_ent, hr2t, rt2h = get_indtest_test_dataset_and_train_g(args)
        self.test_triples = KGEEvalData(args, data['test'], num_ent, hr2t, rt2h)
        if self.args.eval_task == 'link_prediction':
            self.test_triples.num_cand = 50
        elif self.args.eval_task == 'triple_classification':
            self.test_triples.num_cand = 1

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples