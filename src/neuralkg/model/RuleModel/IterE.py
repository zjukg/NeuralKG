import torch.nn as nn
import torch
import os
from .model import Model
from IPython import embed
from collections import defaultdict
import numpy as np
import pickle 
import copy

class IterE(Model):
    """`Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. (WWW'19)`_ (IterE).

    Attributes:
        args: Model configuration parameters.
        epsilon: Caculate embedding_range.
        margin: Caculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
    
    .. _Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. (WWW'19): https://dl.acm.org/doi/10.1145/3308558.3313612
    """

    def __init__(self, args, train_sampler, test_sampler):
        super(IterE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()
        #print(self.args)
        #print(train_sampler)
        #print('run get_axiom()')
        self.train_sampler = train_sampler
        self.train_triples_base = copy.deepcopy(train_sampler.train_triples)
        self.select_probability = self.args.select_probability
        self.max_entialments = self.args.max_entialments
        self.axiom_types  = self.args.axiom_types
        self.axiom_weight = self.args.axiom_weight
        self.inject_triple_percent = self.args.inject_triple_percent
        self.sparsity = 0.995
        
        self.num_entity = self.args.num_ent
        
        self.relation2id=train_sampler.rel2id
        
        self.train_ids=train_sampler.train_triples
        self.valid_ids=train_sampler.valid_triples
        self.test_ids=train_sampler.test_triples
        
        #print(len(self.train_ids))
        #print(len(self.valid_ids))
        #print(len(self.test_ids))
        
        self.train_ids_labels_inject = np.reshape([], [-1, 4])
        
        # generate r_ht, hr_t
        print('# generate r_ht, hr_t')
        self.r_ht, self.hr_t, self.tr_h, self.hr_t_all, self.tr_h_all = self._generate(self.train_ids, self.valid_ids, self.test_ids)
        
        # generate entity2frequency and entity2sparsity dict
        print('# generate entity2frequency and entity2sparsity dict')
        self.entity2frequency, self.entity2sparsity = self._entity2frequency()
    
        print('# get_axiom')
        self.get_axiom()
        
        
        #self.rule, self.conf = self.get_rule(self.relation2id)
        
    def _entity2frequency(self):
        ent2freq = {ent:0 for ent in range(self.num_entity)}
        ent2sparsity = {ent:-1 for ent in range(self.num_entity)}
        for h,r,t in self.train_ids:
            ent2freq[h] += 1
            ent2freq[t] += 1
        ent_freq_list = np.asarray([ent2freq[ent] for ent in range(self.num_entity)])
        ent_freq_list_sort = np.argsort(ent_freq_list)

        max_freq = max(list(ent2freq))
        min_freq = min(list(ent2freq))
        for ent, freq in ent2freq.items():
            sparsity = 1 - (freq-min_freq)/(max_freq - min_freq)
            ent2sparsity[ent] = sparsity
        return ent2freq, ent2sparsity

    
    def _generate(self, train, valid, test):
        r_ht = defaultdict(set)
        hr_t = defaultdict(set)
        tr_h = defaultdict(set)
        hr_t_all = defaultdict(list)
        tr_h_all = defaultdict(list)
        for (h,r,t) in train:
            r_ht[r].add((h,t))
            hr_t[(h,r)].add(t)
            tr_h[(t,r)].add(h)
            hr_t_all[(h,r)].append(t)
            tr_h_all[(t,r)].append(h)
        for (h,r,t) in test+valid:
            hr_t_all[(h,r)].append(t)
            tr_h_all[(t, r)].append(h)
        return r_ht, hr_t, tr_h, hr_t_all, tr_h_all

    def get_axiom(self, ):
        self.axiom_dir = os.path.join(self.args.data_path, 'axiom_pool')
        self.reflexive_dir, self.symmetric_dir, self.transitive_dir, self.inverse_dir, self.subproperty_dir, self.equivalent_dir, self.inferencechain1, self.inferencechain2, self.inferencechain3, self.inferencechain4 = map(lambda x: os.path.join(self.axiom_dir, x),
                                                             ['axiom_reflexive.txt',
                                                              'axiom_symmetric.txt',
                                                              'axiom_transitive.txt',
                                                              'axiom_inverse.txt',
                                                              'axiom_subProperty.txt',
                                                              'axiom_equivalent.txt',
                                                              'axiom_inferenceChain1.txt',
                                                              'axiom_inferenceChain2.txt',
                                                              'axiom_inferenceChain3.txt',
                                                              'axiom_inferenceChain4.txt'])
        # read and materialize axioms
        print('# self._read_axioms()')
        self._read_axioms()
        print('# self._read_axioms()')
        self._materialize_axioms()
        print('# self._read_axioms()')
        self._init_valid_axioms()
        
        
        
        
    def _read_axioms(self):
        # for each axiom, the first id is the basic relation
        self.axiompool_reflexive = self._read_axiompool_file(self.reflexive_dir)
        self.axiompool_symmetric = self._read_axiompool_file(self.symmetric_dir)
        self.axiompool_transitive = self._read_axiompool_file(self.transitive_dir)
        self.axiompool_inverse = self._read_axiompool_file(self.inverse_dir)
        self.axiompool_equivalent = self._read_axiompool_file(self.equivalent_dir)
        self.axiompool_subproperty = self._read_axiompool_file(self.subproperty_dir)
        self.axiompool_inferencechain1 = self._read_axiompool_file(self.inferencechain1)
        self.axiompool_inferencechain2 = self._read_axiompool_file(self.inferencechain2)
        self.axiompool_inferencechain3 = self._read_axiompool_file(self.inferencechain3)
        self.axiompool_inferencechain4 = self._read_axiompool_file(self.inferencechain4)
        self.axiompool = [self.axiompool_reflexive, self.axiompool_symmetric, self.axiompool_transitive,
                          self.axiompool_inverse, self.axiompool_subproperty, self.axiompool_equivalent,
                          self.axiompool_inferencechain1,self.axiompool_inferencechain2,
                          self.axiompool_inferencechain3,self.axiompool_inferencechain4]
    
    def _read_axiompool_file(self, file):
        f = open(file, 'r')
        axioms = []
        for line in f.readlines():
            line_list = line.strip().split('\t')
            axiom_ids = list(map(lambda x: self.relation2id[x], line_list))
            #axiom_ids = self.relation2id[line_list]
            axioms.append(axiom_ids)
        # for the case reflexive pool is empty
        if len(axioms) == 0:
            np.reshape(axioms, [-1, 3])
        return axioms
    
    # for each axioms in axiom pool
    # generate a series of entailments for each axiom
    def _materialize_axioms(self, generate=True, dump=True, load=False):
        if generate:
            self.reflexive2entailment = defaultdict(list)
            self.symmetric2entailment = defaultdict(list)
            self.transitive2entailment = defaultdict(list)
            self.inverse2entailment = defaultdict(list)
            self.equivalent2entailment = defaultdict(list)
            self.subproperty2entailment = defaultdict(list)
            self.inferencechain12entailment = defaultdict(list)
            self.inferencechain22entailment = defaultdict(list)
            self.inferencechain32entailment = defaultdict(list)
            self.inferencechain42entailment = defaultdict(list)

            self.reflexive_entailments, self.reflexive_entailments_num = self._materialize_sparse(self.axiompool_reflexive, type='reflexive')
            self.symmetric_entailments, self.symmetric_entailments_num = self._materialize_sparse(self.axiompool_symmetric, type='symmetric')
            self.transitive_entailments, self.transitive_entailments_num = self._materialize_sparse(self.axiompool_transitive, type='transitive')
            self.inverse_entailments, self.inverse_entailments_num = self._materialize_sparse(self.axiompool_inverse, type='inverse')
            self.subproperty_entailments, self.subproperty_entailments_num = self._materialize_sparse(self.axiompool_subproperty, type='subproperty')
            self.equivalent_entailments, self.equivalent_entailments_num  = self._materialize_sparse(self.axiompool_equivalent, type='equivalent')

            self.inferencechain1_entailments, self.inferencechain1_entailments_num = self._materialize_sparse(self.axiompool_inferencechain1, type='inferencechain1')
            self.inferencechain2_entailments, self.inferencechain2_entailments_num = self._materialize_sparse(self.axiompool_inferencechain2, type='inferencechain2')
            self.inferencechain3_entailments, self.inferencechain3_entailments_num = self._materialize_sparse(self.axiompool_inferencechain3, type='inferencechain3')
            self.inferencechain4_entailments, self.inferencechain4_entailments_num = self._materialize_sparse(self.axiompool_inferencechain4, type='inferencechain4')


            print('reflexive entailments for sparse: ', self.reflexive_entailments_num)
            print('symmetric entailments for sparse: ', self.symmetric_entailments_num)
            print('transitive entailments for sparse: ', self.transitive_entailments_num)
            print('inverse entailments for sparse: ', self.inverse_entailments_num)
            print('subproperty entailments for sparse: ', self.subproperty_entailments_num)
            print('equivalent entailments for sparse: ', self.equivalent_entailments_num)
            print('inferencechain1 entailments for sparse: ', self.inferencechain1_entailments_num)
            print('inferencechain2 entailments for sparse: ', self.inferencechain2_entailments_num)
            print('inferencechain3 entailments for sparse: ', self.inferencechain3_entailments_num)
            print('inferencechain4 entailments for sparse: ', self.inferencechain4_entailments_num)


            print("finish generate axioms entailments for sparse")


        if dump:
            pickle.dump(self.reflexive_entailments, open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'wb'))
            pickle.dump(self.symmetric_entailments, open(os.path.join(self.axiom_dir, 'symmetric_entailments'), 'wb'))
            pickle.dump(self.transitive_entailments, open(os.path.join(self.axiom_dir, 'transitive_entailments'), 'wb'))
            pickle.dump(self.inverse_entailments, open(os.path.join(self.axiom_dir, 'inverse_entailments'), 'wb'))
            pickle.dump(self.subproperty_entailments, open(os.path.join(self.axiom_dir, 'subproperty_entailments'), 'wb'))
            #pickle.dump(self.inferencechain_entailments, open(os.path.join(self.axiom_dir, 'inferencechain_entailments'), 'wb'))
            pickle.dump(self.equivalent_entailments, open(os.path.join(self.axiom_dir, 'equivalent_entailments'), 'wb'))

            pickle.dump(self.inferencechain1_entailments,
                        open(os.path.join(self.axiom_dir, 'inferencechain1_entailments'), 'wb'))
            pickle.dump(self.inferencechain2_entailments,
                        open(os.path.join(self.axiom_dir, 'inferencechain2_entailments'), 'wb'))
            pickle.dump(self.inferencechain3_entailments,
                        open(os.path.join(self.axiom_dir, 'inferencechain3_entailments'), 'wb'))
            pickle.dump(self.inferencechain4_entailments,
                        open(os.path.join(self.axiom_dir, 'inferencechain4_entailments'), 'wb'))

            print("finish dump axioms entialments")

        if load:
            print("load refexive entailments...")
            self.reflexive_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'rb'))
            print(self.reflexive_entailments)
            print('load symmetric entailments...')
            self.symmetric_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'symmetric_entailments'), 'rb'))
            print("load transitive entialments... ")
            self.transitive_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'transitive_entailments'), 'rb'))
            print("load inverse entailments...")
            self.inverse_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'inverse_entailments'), 'rb'))
            print("load subproperty entailments...")
            self.subproperty_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'subproperty_entailments'), 'rb'))
            #print("load inferencechain entailments...")
            #self.inferencechain_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'inferencechain_entailments'), 'rb'))
            print("load equivalent entialments...")
            self.equivalent_entailments = pickle.load(open(os.path.join(self.axiom_dir, 'equivalent_entailments'), 'rb'))

            print("load inferencechain1 entailments...")
            self.inferencechain1_entailments = pickle.load(
                open(os.path.join(self.axiom_dir, 'inferencechain1_entailments'), 'rb'))
            print("load inferencechain2 entailments...")
            self.inferencechain2_entailments = pickle.load(
                open(os.path.join(self.axiom_dir, 'inferencechain2_entailments'), 'rb'))
            print("load inferencechain3 entailments...")
            self.inferencechain3_entailments = pickle.load(
                open(os.path.join(self.axiom_dir, 'inferencechain3_entailments'), 'rb'))
            print("load inferencechain4 entailments...")
            self.inferencechain4_entailments = pickle.load(
                open(os.path.join(self.axiom_dir, 'inferencechain4_entailments'), 'rb'))

            print("finish load axioms entailments")

    def _materialize_sparse(self, axioms, type=None, sparse = False):
        inference = []
        # axiom2entailment is a dict
        # with the all axioms in the axiom pool as keys
        # and all the entailments for each axiom as values
        axiom_list = axioms
        length = len(axioms)
        max_entailments = self.max_entialments
        num = 0
        if length == 0:
            if type == 'reflexive':
                np.reshape(inference, [-1, 3])
            elif type == 'symmetric' or type =='inverse' or  type =='equivalent' or type =='subproperty':
                np.reshape(inference, [-1, 6])
            elif type=='transitive' or type=='inferencechain':
                np.reshape(inference, [-1, 9])
            else:
                raise NotImplementedError
            return inference, num

        if type == 'reflexive':
            for axiom in axiom_list:
                axiom_key =tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h,t) in self.r_ht[r]:
                    # filter the axiom with too much entailments
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 2])
                        break
                    if h != t and self.entity2sparsity[h]>self.sparsity:
                        num += 1
                        inference_tmp.append([h,r,h])

                for entailment in inference_tmp:
                    self.reflexive2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'symmetric':
            #self.symmetric2entailment = defaultdict(list)
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h,t) in self.r_ht[r]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 2])
                        break
                    if (t,h) not in self.r_ht[r] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
                        num += 1
                        inference_tmp.append([h,r,t,t,r,h])


                for entailment in inference_tmp:
                    self.symmetric2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'transitive':
            #self.transitive2entailment = defaultdict(list)
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h,t) in self.r_ht[r]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    # (t,r,e) exist but (h,r,e) not exist and e!=h
                    for e in self.hr_t[(t,r)]- self.hr_t[(h,r)]:
                        if e != h and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[e]>self.sparsity):
                            num += 1
                            inference_tmp.append([h,r,t,t,r,e,h,r,e])

                for entailment in inference_tmp:
                    self.transitive2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inverse':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r1,r2 = axiom
                inference_tmp = []
                for (h,t) in self.r_ht[r1]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 6])
                        break
                    if (t,h) not in self.r_ht[r2] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
                        num += 1
                        inference_tmp.append([h,r1,t, t,r2,h])
                        #self.inverse2entailment[axiom_key].append([h,r1,t, t,r2,h])

                for entailment in inference_tmp:
                    self.inverse2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'equivalent' or type =='subproperty':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r1,r2 = axiom
                inference_tmp = []
                for (h,t) in self.r_ht[r1]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 6])
                        break
                    if (h,t) not in self.r_ht[r2] and (self.entity2sparsity[h]>self.sparsity or self.entity2sparsity[t]>self.sparsity):
                        num += 1
                        inference_tmp.append([h,r1,t, h,r2,t])

                for entailment in inference_tmp:
                    self.equivalent2entailment[axiom_key].append(entailment)
                    self.subproperty2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)


        if type == 'inferencechain1':
            self.inferencechain12entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (e, h) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.hr_t[(e, r3)]:
                        if (h, t) not in self.r_ht[r1] and (
                                        self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
                            num += 1
                            inference_tmp.append([e, r2, h, e, r3, t, h, r1, t])
                            #self.inferencechain12entailment[axiom_key].append([[e, r2, h, e, r3, t, h, r1, t]])


                for entailment in inference_tmp:
                    self.inferencechain12entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain2':
            self.inferencechain22entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (e, h) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.tr_h[(e, r3)]:
                        if (h, t) not in self.r_ht[r1] and (
                                        self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
                            num += 1
                            inference_tmp.append([e, r2, h, t, r3, e, h, r1, t])
                            #self.inferencechain22entailment[axiom_key].append([[e, r2, h, t, r3, e, h, r1, t]])

                for entailment in inference_tmp:
                    self.inferencechain22entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)


        if type == 'inferencechain3':
            self.inferencechain32entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (h, e) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.hr_t[(e, r3)]:
                        if (h, t) not in self.r_ht[r1] and (
                                        self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
                            num += 1
                            inference_tmp.append([h, r2, e, e, r3, t, h, r1, t])


                for entailment in inference_tmp:
                    self.inferencechain32entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain4':
            self.inferencechain42entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (h, e) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.tr_h[(e, r3)]:
                        if (h, t) not in self.r_ht[r1] and (
                                        self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
                            num += 1
                            inference_tmp.append([h, r2, e, t, r3, e, h, r1, t])

                for entailment in inference_tmp:
                    self.inferencechain42entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)
        return inference, num

    def _materialize(self, axioms, type=None, sparse=False):
        inference = []
        # axiom2entailment is a dict
        # with the all axioms in the axiom pool as keys
        # and all the entailments for each axiom as values
        axiom_list = axioms
        # print('axiom_list', axiom_list)
        length = len(axioms)
        max_entailments = 5000
        num = 0
        if length == 0:
            if type == 'reflexive':
                np.reshape(inference, [-1, 3])
            elif type == 'symmetric' or type == 'inverse' or type == 'equivalent' or type == 'subproperty':
                np.reshape(inference, [-1, 6])
            elif type == 'transitive' or type == 'inferencechain':
                np.reshape(inference, [-1, 9])
            else:
                raise NotImplementedError
            return inference, num

        if type == 'reflexive':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h, t) in self.r_ht[r]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 2])
                        break
                    if h != t: #and self.entity2sparsity[h] > self.sparsity:
                        num += 1
                        inference_tmp.append([h, r, h])
                for entailment in inference_tmp:
                    self.reflexive2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'symmetric':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h, t) in self.r_ht[r]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 2])
                        break
                    if (t, h) not in self.r_ht[r]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
                        num += 1
                        inference_tmp.append([h, r, t, t, r, h])
                for entailment in inference_tmp:
                    self.symmetric2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'transitive':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r = axiom[0]
                inference_tmp = []
                for (h, t) in self.r_ht[r]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    # (t,r,e) exist but (h,r,e) not exist and e!=h
                    for e in self.hr_t[(t, r)] - self.hr_t[(h, r)]:
                        if e != h: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[e] > self.sparsity):
                            num += 1
                            inference_tmp.append([h, r, t, t, r, e, h, r, e])

                for entailment in inference_tmp:
                    self.transitive2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inverse':
            # self.inverse2entailment = defaultdict(list)
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r1, r2 = axiom
                inference_tmp = []
                for (h, t) in self.r_ht[r1]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 6])
                        break
                    if (t, h) not in self.r_ht[r2]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
                        num += 1
                        inference_tmp.append([h, r1, t, t, r2, h])
                for entailment in inference_tmp:
                    self.inverse2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'equivalent' or type == 'subproperty':
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                r1, r2 = axiom
                inference_tmp = []
                for (h, t) in self.r_ht[r1]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 6])
                        break
                    if (h, t) not in self.r_ht[r2]: #and (self.entity2sparsity[h] > self.sparsity or self.entity2sparsity[t] > self.sparsity):
                        num += 1
                        inference_tmp.append([h, r1, t, h, r2, t])

                for entailment in inference_tmp:
                    self.equivalent2entailment[axiom_key].append(entailment)
                    self.subproperty2entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain1':
            self.inferencechain12entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (e, h) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.hr_t[(e, r3)]:
                        if (h, t) not in self.r_ht[r1]:
                            num += 1
                            inference_tmp.append([e, r2, h, e, r3, t, h, r1, t])
                for entailment in inference_tmp:
                    self.inferencechain12entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain2':
            self.inferencechain22entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (e, h) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.tr_h[(e, r3)]:
                        if (h, t) not in self.r_ht[r1]:
                            num += 1
                            inference_tmp.append([e, r2, h, t, r3, e, h, r1, t])
                for entailment in inference_tmp:
                    self.inferencechain22entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain3':
            self.inferencechain32entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                # print('%d/%d' % (i, length))
                r1, r2, r3 = axiom
                inference_tmp = []
                for (h, e) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.hr_t[(e, r3)]:
                        if (h, t) not in self.r_ht[r1]:
                            num += 1
                            inference_tmp.append([h, r2, e, e, r3, t, h, r1, t])
                for entailment in inference_tmp:
                    self.inferencechain32entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)

        if type == 'inferencechain4':
            self.inferencechain42entailment = defaultdict(list)
            i = 0
            for axiom in axiom_list:
                axiom_key = tuple(axiom)
                i += 1
                r1, r2, r3 = axiom
                inference_tmp = []
                for (h, e) in self.r_ht[r2]:
                    if len(inference_tmp) > max_entailments:
                        inference_tmp = np.reshape([], [-1, 9])
                        break
                    for t in self.tr_h[(e, r3)]:
                        if (h, t) not in self.r_ht[r1]:
                            num += 1
                            inference_tmp.append([h, r2, e, t, r3, e, h, r1, t])
                for entailment in inference_tmp:
                    self.inferencechain42entailment[axiom_key].append(entailment)
                inference.append(inference_tmp)
        return inference, num
    
    def _init_valid_axioms(self):
        # init valid axioms
        self.valid_reflexive, self.valid_symmetric, self.valid_transitive,\
        self.valid_inverse, self.valid_subproperty, self.valid_equivalent,\
        self.valid_inferencechain1, self.valid_inferencechain2, \
        self.valid_inferencechain3, self.valid_inferencechain4 = [[] for x in range(self.axiom_types)]

        # init valid axiom entailments
        self.valid_reflexive2entailment, self.valid_symmetric2entailment, self.valid_transitive2entailment, \
        self.valid_inverse2entailment, self.valid_subproperty2entailment, self.valid_equivalent2entailment, \
        self.valid_inferencechain12entailment, self.valid_inferencechain22entailment, \
        self.valid_inferencechain32entailment, self.valid_inferencechain42entailment = [[] for x in range(self.axiom_types)]

        # init valid axiom entailments probability
        self.valid_reflexive_p, self.valid_symmetric_p, self.valid_transitive_p, \
        self.valid_inverse_p, self.valid_subproperty_p, self.valid_equivalent_p, \
        self.valid_inferencechain1_p, self.valid_inferencechain2_p,\
        self.valid_inferencechain3_p, self.valid_inferencechain4_p= [[] for x in range(self.axiom_types)]

        # init valid axiom batchsize
        self.reflexive_batchsize = 1
        self.symmetric_batchsize = 1
        self.transitive_batchsize = 1
        self.inverse_batchsize = 1
        self.subproperty_batchsize = 1
        self.equivalent_batchsize = 1
        #self.inferencechain_batchsize = 1
        self.inferencechain1_batchsize = 1
        self.inferencechain2_batchsize = 1
        self.inferencechain3_batchsize = 1
        self.inferencechain4_batchsize = 1


    # add the new triples from axioms to training triple
    def update_train_triples(self, epoch=0, update_per = 10):
        """add the new triples from axioms to training triple

        Args:
            epoch (int, optional): epoch in training process. Defaults to 0.
            update_per (int, optional): Defaults to 10.

        Returns:
            updated_train_data: training triple after adding the new triples from axioms
        """
        reflexive_triples, symmetric_triples, transitive_triples, inverse_triples,\
            equivalent_triples, subproperty_triples, inferencechain1_triples, \
            inferencechain2_triples, inferencechain3_triples, inferencechain4_triples = [ np.reshape(np.asarray([]), [-1, 3]) for i in range(self.axiom_types)]
        reflexive_p, symmetric_p, transitive_p, inverse_p, \
            equivalent_p, subproperty_p, inferencechain1_p, \
            inferencechain2_p, inferencechain3_p, inferencechain4_p = [np.reshape(np.asarray([]), [-1, 1]) for i in
                                                                           range(self.axiom_types)]
        updated_train_data=None
        if epoch >= 5:
            print("len(self.valid_reflexive2entailment):", len(self.valid_reflexive2entailment))
            print("len(self.valid_symmetric2entailment):", len(self.valid_symmetric2entailment))
            print("len(self.valid_transitive2entailment)", len(self.valid_transitive2entailment))
            print("len(self.valid_inverse2entailment)", len(self.valid_inverse2entailment))
            print("len(self.valid_equivalent2entailment)", len(self.valid_equivalent2entailment))
            print("len(self.valid_subproperty2entailment)", len(self.valid_subproperty2entailment))

            valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment,\
            valid_inverse2entailment, valid_equivalent2entailment, valid_subproperty2entailment, \
            valid_inferencechain12entailment, valid_inferencechain22entailment,\
            valid_inferencechain32entailment, valid_inferencechain42entailment = [[] for i in range(10)]

            if len(self.valid_reflexive2entailment)>0:
                valid_reflexive2entailment = np.reshape(np.asarray(self.valid_reflexive2entailment), [-1, 3])
                reflexive_triples = np.asarray(valid_reflexive2entailment)[:, -3:]
                reflexive_p = np.reshape(np.asarray(self.valid_reflexive_p),[-1,1])

            if len(self.valid_symmetric2entailment) > 0:
                valid_symmetric2entailment = np.reshape(np.asarray(self.valid_symmetric2entailment), [-1, 6])
                symmetric_triples = np.asarray(valid_symmetric2entailment)[:, -3:]
                symmetric_p = np.reshape(np.asarray(self.valid_symmetric_p),[-1,1])

            if len(self.valid_transitive2entailment) > 0:
                valid_transitive2entailment = np.reshape(np.asarray(self.valid_transitive2entailment), [-1, 9])
                transitive_triples = np.asarray(valid_transitive2entailment)[:, -3:]
                transitive_p = np.reshape(np.asarray(self.valid_transitive_p), [-1, 1])

            if len(self.valid_inverse2entailment) > 0:
                valid_inverse2entailment = np.reshape(np.asarray(self.valid_inverse2entailment), [-1, 6])
                inverse_triples = np.asarray(valid_inverse2entailment)[:, -3:]
                inverse_p = np.reshape(np.asarray(self.valid_inverse_p), [-1, 1])

            if len(self.valid_equivalent2entailment) > 0:
                valid_equivalent2entailment = np.reshape(np.asarray(self.valid_equivalent2entailment), [-1, 6])
                equivalent_triples = np.asarray(valid_equivalent2entailment)[:, -3:]
                equivalent_p = np.reshape(np.asarray(self.valid_equivalent_p), [-1, 1])

            if len(self.valid_subproperty2entailment) > 0:
                valid_subproperty2entailment = np.reshape(np.asarray(self.valid_subproperty2entailment), [-1, 6])
                subproperty_triples = np.asarray(valid_subproperty2entailment)[:, -3:]
                subproperty_p = np.reshape(np.asarray(self.valid_subproperty_p),[-1,1])

            if len(self.valid_inferencechain12entailment) > 0:
                valid_inferencechain12entailment = np.reshape(np.asarray(self.valid_inferencechain12entailment), [-1, 9])
                inferencechain1_triples = np.asarray(valid_inferencechain12entailment)[:, -3:]
                inferencechain1_p = np.reshape(np.asarray(self.valid_inferencechain1_p), [-1, 1])

            if len(self.valid_inferencechain22entailment) > 0:
                valid_inferencechain22entailment = np.reshape(np.asarray(self.valid_inferencechain22entailment), [-1, 9])
                inferencechain2_triples = np.asarray(valid_inferencechain22entailment)[:, -3:]
                inferencechain2_p = np.reshape(np.asarray(self.valid_inferencechain2_p), [-1, 1])

            if len(self.valid_inferencechain32entailment) > 0:
                valid_inferencechain32entailment = np.reshape(np.asarray(self.valid_inferencechain32entailment), [-1, 9])
                inferencechain3_triples = np.asarray(valid_inferencechain32entailment)[:, -3:]
                inferencechain3_p = np.reshape(np.asarray(self.valid_inferencechain3_p), [-1, 1])

            if len(self.valid_inferencechain42entailment) > 0:
                valid_inferencechain42entailment = np.reshape(np.asarray(self.valid_inferencechain42entailment), [-1, 9])
                inferencechain4_triples = np.asarray(valid_inferencechain42entailment)[:, -3:]
                inferencechain4_p = np.reshape(np.asarray(self.valid_inferencechain4_p), [-1, 1])

            # pickle.dump(self.reflexive_entailments, open(os.path.join(self.axiom_dir, 'reflexive_entailments'), 'wb'))
            # store all the injected triples
            entailment_all = (valid_reflexive2entailment, valid_symmetric2entailment, valid_transitive2entailment,
                     valid_inverse2entailment, valid_equivalent2entailment, valid_subproperty2entailment,
                     valid_inferencechain12entailment,valid_inferencechain22entailment,
                              valid_inferencechain32entailment,valid_inferencechain42entailment)
            pickle.dump(entailment_all, open(os.path.join(self.axiom_dir, 'valid_entailments.pickle'), 'wb'))



        train_inject_triples = np.concatenate([reflexive_triples, symmetric_triples, transitive_triples, inverse_triples,
                                                equivalent_triples, subproperty_triples, inferencechain1_triples,
                                               inferencechain2_triples,inferencechain3_triples,inferencechain4_triples],
                                                axis=0)

        train_inject_triples_p = np.concatenate([reflexive_p,symmetric_p, transitive_p, inverse_p,
                                               equivalent_p, subproperty_p, inferencechain1_p,
                                                 inferencechain2_p,inferencechain3_p,inferencechain4_p],
                                              axis=0)

        self.train_inject_triples = train_inject_triples
        inject_labels = np.reshape(np.ones(len(train_inject_triples)), [-1, 1]) * self.axiom_weight * train_inject_triples_p
        train_inject_ids_labels = np.concatenate([train_inject_triples, inject_labels],
                                                axis=1)


        self.train_ids_labels_inject = train_inject_triples#train_inject_ids_labels


        print('num reflexive triples', len(reflexive_triples))
        print('num symmetric triples', len(symmetric_triples))
        print('num transitive triples', len(transitive_triples))
        print('num inverse triples', len(inverse_triples))
        print('num equivalent triples', len(equivalent_triples))
        print('num subproperty triples', len(subproperty_triples))
        print('num inferencechain1 triples', len(inferencechain1_triples))
        print('num inferencechain2 triples', len(inferencechain2_triples))
        print('num inferencechain3 triples', len(inferencechain3_triples))
        print('num inferencechain4 triples', len(inferencechain4_triples))
        #print(self.train_ids_labels_inject)
        updated_train_data=self.generate_new_train_triples()
        return updated_train_data

    def split_embedding(self, embedding):
        """split embedding

        Args:
            embedding: embeddings need to be splited, shape:[None, dim].

        Returns:
            probability: The similrity between two matrices.
        """
        # embedding: [None, dim]
        assert self.args.emb_dim % 4 == 0
        num_scalar = self.args.emb_dim // 2
        num_block = self.args.emb_dim // 4
        if len(embedding.size()) ==2:
            embedding_scalar = embedding[:, 0:num_scalar]
            embedding_x = embedding[:, num_scalar:-num_block]
            embedding_y = embedding[:, -num_block:]
        elif len(embedding.size()) ==3:
            embedding_scalar = embedding[:, :, 0:num_scalar]
            embedding_x = embedding[:, :, num_scalar:-num_block]
            embedding_y = embedding[:, :, -num_block:]
        else:
            raise NotImplementedError

        return embedding_scalar, embedding_x, embedding_y

    
    # calculate the similrity between two matrices
    # head: [?, dim]
    # tail: [?, dim] or [1,dim]
    def sim(self, head=None, tail=None, arity=None):
        """calculate the similrity between two matrices

        Args:
            head: embeddings of head, shape:[batch_size, dim].
            tail: embeddings of tail, shape:[batch_size, dim] or [1, dim].
            arity: 1，2 or 3

        Returns:
            probability: The similrity between two matrices.

        """
        if arity == 1:
            A_scalar, A_x, A_y = self.split_embedding(head)
        elif arity == 2:
            M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
            M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
            A_scalar= M1_scalar * M2_scalar
            A_x = M1_x*M2_x - M1_y*M2_y
            A_y = M1_x*M2_y + M1_y*M2_x
        elif arity==3:
            M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
            M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
            M3_scalar, M3_x, M3_y = self.split_embedding(head[2])
            M1M2_scalar = M1_scalar * M2_scalar
            M1M2_x = M1_x * M2_x - M1_y * M2_y
            M1M2_y = M1_x * M2_y + M1_y * M2_x
            A_scalar = M1M2_scalar * M3_scalar
            A_x = M1M2_x * M3_x - M1M2_y * M3_y
            A_y = M1M2_x * M3_y + M1M2_y * M3_x
        else:
            raise NotImplemented
        B_scala, B_x, B_y = self.split_embedding(tail)

        similarity = torch.cat([(A_scalar - B_scala)**2, (A_x - B_x)**2, (A_x - B_x)**2, (A_y - B_y)**2, (A_y - B_y)**2 ], dim=1)
        similarity = torch.sqrt(torch.sum(similarity, dim=1))

        #recale the probability
        probability = (torch.max(similarity)-similarity)/(torch.max(similarity)-torch.min(similarity))

        return probability

    # generate a probality for each axiom in axiom pool
    def run_axiom_probability(self):
        """this function is used to generate a probality for each axiom in axiom pool

        """
        self.identity = torch.cat((torch.ones(int(self.args.emb_dim-self.args.emb_dim/4)),torch.zeros(int(self.args.emb_dim/4))),0).unsqueeze(0).cuda()
        
        if len(self.axiompool_reflexive) != 0: 
            index = torch.LongTensor(self.axiompool_reflexive).cuda()
            reflexive_embed = self.rel_emb(index)
            reflexive_prob = self.sim(head=reflexive_embed[:, 0, :], tail=self.identity, arity=1)
        else: 
            reflexive_prob = []

        if len(self.axiompool_symmetric) != 0: 
            index = torch.LongTensor(self.axiompool_symmetric).cuda()
            symmetric_embed = self.rel_emb(index)
            symmetric_prob = self.sim(head=[symmetric_embed[:, 0, :], symmetric_embed[:, 0, :]], tail=self.identity, arity=2)
            #symmetric_prob = sess.run(self.symmetric_probability, {self.symmetric_pool: self.axiompool_symmetric})
        else: 
            symmetric_prob = []

        if len(self.axiompool_transitive) != 0: 
            index = torch.LongTensor(self.axiompool_transitive).cuda()
            transitive_embed = self.rel_emb(index)
            transitive_prob = self.sim(head=[transitive_embed[:, 0, :], transitive_embed[:, 0, :]], tail=transitive_embed[:, 0, :], arity=2)
            #transitive_prob = sess.run(self.transitive_probability, {self.transitive_pool: self.axiompool_transitive})
        else: 
            transitive_prob = []

        if len(self.axiompool_inverse) != 0: 
            index = torch.LongTensor(self.axiompool_inverse).cuda()
            #inverse_prob = sess.run(self.inverse_probability, {self.inverse_pool: self.axiompool_inverse})
            inverse_embed = self.rel_emb(index)
            inverse_probability1 = self.sim(head=[inverse_embed[:, 0,:],inverse_embed[:, 1,:]], tail = self.identity, arity=2)
            inverse_probability2 = self.sim(head=[inverse_embed[:,1,:],inverse_embed[:, 0,:]], tail=self.identity, arity=2)
            inverse_prob = (inverse_probability1 + inverse_probability2)/2
        else: 
            inverse_prob = []

        if len(self.axiompool_subproperty) != 0: 
            index = torch.LongTensor(self.axiompool_subproperty).cuda()
            #subproperty_prob = sess.run(self.subproperty_probability, {self.subproperty_pool: self.axiompool_subproperty})
            subproperty_embed = self.rel_emb(index)
            subproperty_prob = self.sim(head=subproperty_embed[:, 0,:], tail=subproperty_embed[:, 1, :], arity=1)
        else: 
            subproperty_prob = []

        if len(self.axiompool_equivalent) != 0: 
            index = torch.LongTensor(self.axiompool_equivalent).cuda()
            #equivalent_prob = sess.run(self.equivalent_probability, {self.equivalent_pool: self.axiompool_equivalent})
            equivalent_embed = self.rel_emb(index)
            equivalent_prob = self.sim(head=equivalent_embed[:, 0,:], tail=equivalent_embed[:, 1,:], arity=1)
        else: 
            equivalent_prob = []

        if len(self.axiompool_inferencechain1) != 0:
            index = torch.LongTensor(self.axiompool_inferencechain1).cuda()
            inferencechain_embed = self.rel_emb(index) 
            inferencechain1_prob = self.sim(head=[inferencechain_embed[:, 1, :], inferencechain_embed[:, 0, :]], tail=inferencechain_embed[:, 2, :], arity=2)
        else:
            inferencechain1_prob = []

        if len(self.axiompool_inferencechain2) != 0:
            index = torch.LongTensor(self.axiompool_inferencechain2).cuda()
            inferencechain_embed = self.rel_emb(index)
            inferencechain2_prob = self.sim(head=[inferencechain_embed[:, 2, :], inferencechain_embed[:, 1, :], inferencechain_embed[:, 0, :]], tail=self.identity, arity=3)
        else:
            inferencechain2_prob = []

        if len(self.axiompool_inferencechain3) != 0:
            index = torch.LongTensor(self.axiompool_inferencechain3).cuda()
            inferencechain_embed = self.rel_emb(index)
            inferencechain3_prob = self.sim(head=[inferencechain_embed[:, 1, :], inferencechain_embed[:, 2, :]], tail=inferencechain_embed[:, 0, :], arity=2)
        else:
            inferencechain3_prob = []

        if len(self.axiompool_inferencechain4) != 0:
            index = torch.LongTensor(self.axiompool_inferencechain4).cuda()
            inferencechain_embed = self.rel_emb(index)
            inferencechain4_prob = self.sim(head=[inferencechain_embed[:, 0, :], inferencechain_embed[:, 2, :]],tail=inferencechain_embed[:, 1, :], arity=2)
        else:
            inferencechain4_prob = []

        output = [reflexive_prob, symmetric_prob, transitive_prob, inverse_prob,
                  subproperty_prob,equivalent_prob,inferencechain1_prob, inferencechain2_prob,
                  inferencechain3_prob, inferencechain4_prob]
        return output

    def update_valid_axioms(self, input):
        """this function is used to select high probability axioms as valid axioms and record their scores

        """
        # 
        # 
        valid_axioms = [self._select_high_probability(list(prob), axiom) for prob,axiom in zip(input, self.axiompool)]

        self.valid_reflexive, self.valid_symmetric, self.valid_transitive, \
        self.valid_inverse, self.valid_subproperty, self.valid_equivalent, \
        self.valid_inferencechain1, self.valid_inferencechain2, \
        self.valid_inferencechain3, self.valid_inferencechain4 = valid_axioms
        # update the batchsize of axioms and entailments
        self._reset_valid_axiom_entailment()


    def _select_high_probability(self, prob, axiom):
        # select the high probability axioms and recore their probabilities
        valid_axiom = [[axiom[prob.index(p)],[p]] for p in prob if p>self.select_probability]
        return valid_axiom

    def _reset_valid_axiom_entailment(self):

        self.infered_hr_t = defaultdict(set)
        self.infered_tr_h = defaultdict(set)

        self.valid_reflexive2entailment, self.valid_reflexive_p = \
            self._valid_axiom2entailment(self.valid_reflexive, self.reflexive2entailment)

        self.valid_symmetric2entailment, self.valid_symmetric_p = \
            self._valid_axiom2entailment(self.valid_symmetric, self.symmetric2entailment)

        self.valid_transitive2entailment, self.valid_transitive_p = \
            self._valid_axiom2entailment(self.valid_transitive, self.transitive2entailment)

        self.valid_inverse2entailment, self.valid_inverse_p = \
            self._valid_axiom2entailment(self.valid_inverse, self.inverse2entailment)

        self.valid_subproperty2entailment, self.valid_subproperty_p = \
            self._valid_axiom2entailment(self.valid_subproperty, self.subproperty2entailment)

        self.valid_equivalent2entailment, self.valid_equivalent_p = \
            self._valid_axiom2entailment(self.valid_equivalent, self.equivalent2entailment)


        self.valid_inferencechain12entailment, self.valid_inferencechain1_p = \
            self._valid_axiom2entailment(self.valid_inferencechain1, self.inferencechain12entailment)

        self.valid_inferencechain22entailment, self.valid_inferencechain2_p = \
            self._valid_axiom2entailment(self.valid_inferencechain2, self.inferencechain22entailment)

        self.valid_inferencechain32entailment, self.valid_inferencechain3_p = \
            self._valid_axiom2entailment(self.valid_inferencechain3, self.inferencechain32entailment)
        self.valid_inferencechain42entailment, self.valid_inferencechain4_p = \
            self._valid_axiom2entailment(self.valid_inferencechain4, self.inferencechain42entailment)


    def _valid_axiom2entailment(self, valid_axiom, axiom2entailment):
        valid_axiom2entailment = []
        valid_axiom_p = []
        for axiom_p in valid_axiom:
            axiom = tuple(axiom_p[0])
            p = axiom_p[1]
            for entailment in axiom2entailment[axiom]:
                valid_axiom2entailment.append(entailment)
                valid_axiom_p.append(p)
                h,r,t = entailment[-3:]
                self.infered_hr_t[(h,r)].add(t)
                self.infered_tr_h[(t,r)].add(h)
        return valid_axiom2entailment, valid_axiom_p


    # updata new train triples:
    def generate_new_train_triples(self):
        """The function is to updata new train triples and used after each training epoch end

        Returns:
            self.train_sampler.train_triples: The new training dataset (triples).
        """
        self.train_sampler.train_triples = copy.deepcopy(self.train_triples_base)
        print('generate_new_train_triples...')
        #origin_triples = train_sampler.train_triples
        inject_triples = self.train_ids_labels_inject
        inject_num = int(self.inject_triple_percent*len(self.train_sampler.train_triples))
        if len(inject_triples)> inject_num and inject_num >0:
            np.random.shuffle(inject_triples)
            inject_triples = inject_triples[:inject_num]
        #train_triples = np.concatenate([origin_triples, inject_triples], axis=0)
        print('当前train_sampler.train_triples数目',len(self.train_sampler.train_triples))
        
        for h,r,t in inject_triples:       
            self.train_sampler.train_triples.append((int(h),int(r),int(t)))
        print('添加后train_sampler.train_triples数目',len(self.train_sampler.train_triples))

        return self.train_sampler.train_triples
    
    def get_rule(self, rel2id):
        """Get rule for rule_base KGE models, such as ComplEx_NNE model.
        Get rule and confidence from _cons.txt file.
        Update:
            (rule_p, rule_q): Rule.
            confidence: The confidence of rule.
        """
        rule_p, rule_q, confidence = [], [], []
        with open(os.path.join(self.args.data_path, '_cons.txt')) as file:
            lines = file.readlines()
            for line in lines:
                rule_str, trust = line.strip().split()
                body, head = rule_str.split(',')
                if '-' in body:
                    rule_p.append(rel2id[body[1:]])
                    rule_q.append(rel2id[head])
                else:
                    rule_p.append(rel2id[body])
                    rule_q.append(rel2id[head])
                confidence.append(float(trust))
        rule_p = torch.tensor(rule_p).cuda()
        rule_q = torch.tensor(rule_q).cuda()
        confidence = torch.tensor(confidence).cuda()
        return (rule_p, rule_q), confidence

    """def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())"""
    
    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

        

        
    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is DistMult.

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        h_scalar, h_x ,h_y = self.split_embedding(head_emb)
        r_scalar, r_x, r_y = self.split_embedding(relation_emb)
        t_scalar, t_x, t_y = self.split_embedding(tail_emb)
        score_scalar = torch.sum(h_scalar * r_scalar * t_scalar, axis=-1)
        score_block = torch.sum(h_x * r_x * t_x
										+ h_x * r_y * t_y
										+ h_y * r_x * t_y
										- h_y * r_y * t_x, axis=-1)
        score = score_scalar + score_block
        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score
    
    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
