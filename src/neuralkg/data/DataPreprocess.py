import numpy as np
from torch.utils.data import Dataset
import torch
import os
from collections import defaultdict as ddict
from IPython import embed


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

    # TODO:把里面的函数再分一分，最基础的部分再初始化的使用调用，其他函数具体情况再调用
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
        """Get entity/relation id, and entity/relation number.

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
        """Get triples id, save in the format of (h, r, t).

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
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

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
        '''Get frequency of a partial triple like (head, relation) or (relation, tail).
        
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
        """Get the set of h2rt and t2hr from train dataset, the data type is numpy.

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
        """Get entity/relation/reverse relation id, and entity/relation number.

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