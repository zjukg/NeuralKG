import os
import dgl
import lmdb
import time
import yaml
import json
import torch
import pickle
import struct
import random
import logging
import datetime
import importlib
import numpy as np
import networkx as nx
import scipy.sparse as ssp
import multiprocessing as mp
from tqdm import tqdm
from scipy.special import softmax
from scipy.sparse import csc_matrix
from collections import defaultdict as ddict
from torch.nn import Parameter
from torch.nn.init import xavier_normal_


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'model.TransE'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def save_config(args):
    args.save_config = False  #防止和load_config冲突，导致把加载的config又保存了一遍
    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime(str(args.model_name)+"_"+str(args.dataset_name)) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.makedirs(os.path.join("config", day_name))
    config = vars(args)
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

def load_config(args, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)
    return args

def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param 

def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))

def deserialize_RMPI(data): 
    data_tuple = pickle.loads(data)
    keys = ('en_nodes', 'r_label', 'g_label', 'en_n_labels', 'dis_nodes', 'dis_n_labels')
    return dict(zip(keys, data_tuple))

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    date_file = os.path.join(args.save_path, date)

    if not os.path.exists(date_file):
        os.makedirs(date_file)

    hour = str(int(dt.strftime("%H")) + 8)
    name = hour + dt.strftime("_%M_%S")
    if args.special_name != None:
        name = args.special_name
    log_file = os.path.join(date_file,  "_".join([args.model_name, args.dataset_name, name, 'train.log']))

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.INFO,
        datefmt='%m-%d %H:%M',
        filename=log_file,
        filemode='a'
    )

def log_metrics(epoch, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s: %.4f at epoch %d' % (metric, metrics[metric], epoch))

def log_step_metrics(step, metrics):
    '''
    Print the evaluation logs for check_per_step
    '''
    for metric in metrics:
        logging.info('%s: %.4f at step %d' % (metric, metrics[metric], step))

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx

def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    rel_reidx = dict()
    relidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(rel_reidx), dict(ent_reidx)

def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(ent_reidx)

def data2pkl(dataset_name):
    '''Store data in pickle'''
    train_tri = []
    file = open('./dataset/{}/train.txt'.format(dataset_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    valid_tri = []
    file = open('./dataset/{}/valid.txt'.format(dataset_name))
    valid_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    test_tri = []
    file = open('./dataset/{}/test.txt'.format(dataset_name))
    test_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    file = open('./dataset/{}_ind/train.txt'.format(dataset_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('./dataset/{}_ind/valid.txt'.format(dataset_name))
    ind_valid_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('./dataset/{}_ind/test.txt'.format(dataset_name))
    ind_test_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri,
                                 'rel2idx': fix_rel_reidx, 'ent2idx': ent_reidx},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri,
                                    'rel2idx': fix_rel_reidx, 'ent2idx': ent_reidx_ind}}

    pickle.dump(save_data, open(f'./dataset/{dataset_name}.pkl', 'wb'))
    
def gen_subgraph_datasets(args, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test' in splits
    if testing:
        adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel = load_ind_data_grail(args)
    else:
        adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, _, _, _, _ = load_data_grail(args)

    graphs = {}
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': args.max_links}

    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], args.num_neg_samples_per_link,
                                                max_size=split['max_size'], constrained_neg_prob=args.constrained_neg_prob)

    links2subgraphs(adj_list, graphs, args, max_label_value, testing)

def load_ind_data_grail(args):
    data = pickle.load(open(args.pk_path, 'rb'))

    splits = ['train', 'test']

    triplets = {}
    for split_name in splits:
        triplets[split_name] = np.array(data['ind_test_graph'][split_name])[:, [0, 2, 1]]

    train_rel2idx = data['ind_test_graph']['rel2idx']
    train_ent2idx = data['ind_test_graph']['ent2idx']
    train_idx2rel = {i: r for r, i in train_rel2idx.items()}
    train_idx2ent = {i: e for e, i in train_ent2idx.items()}

    adj_list = []
    for i in range(len(train_rel2idx)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                shape=(len(train_ent2idx), len(train_ent2idx))))

    return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel 

def load_data_grail(args, add_traspose_rels=False):
    data = pickle.load(open(args.pk_path, 'rb'))

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
    if args.model_name == 'SNRI':
        # Construct the the neighbor relations of each entity
        num_rels = len(train_idx2rel)
        num_ents = len(train_idx2ent)
        h2r = {}
        h2r_len = {}
        t2r = {}
        t2r_len = {}
        
        for triplet in triplets['train']:
            h, t, r = triplet
            if h not in h2r:
                h2r_len[h] = 1
                h2r[h] = [r]
            else:
                h2r_len[h] += 1
                h2r[h].append(r)
            
            if args.add_traspose_rels:
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
        logging.info("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)
        
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
        if args.sort_data:
            triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]

    adj_list = []
    for i in range(len(train_rel2idx)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                shape=(len(train_ent2idx), len(train_ent2idx))))

    return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, h2r, m_h2r, t2r, m_t2r

def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        if params.model_name == 'RMPI':
            en_nodes, en_n_labels, subgraph_size, enc_ratio, num_pruned_nodes, dis_nodes, dis_n_labels = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
            datum = {'en_nodes': en_nodes, 'r_label': r_label, 'g_label': 0, 'en_n_labels': en_n_labels, 'dis_nodes':dis_nodes, 'dis_n_labels':dis_n_labels}
        else:
            nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes, _, _ = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
            datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size

def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)

def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges

def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)

def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value

def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_

    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes, dis_nodes, dis_n_labels = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])
        dis_n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in dis_n_labels])
    if params_.model_name == 'RMPI':
        datum = {'en_nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'en_n_labels': n_labels, 'dis_nodes':dis_nodes, 'dis_n_labels':dis_n_labels}
    else:
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def links2subgraphs(A, graphs, params, max_label_value=None, testing=False):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []
    if params.model_name == 'RMPI':
        BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 10
    else:
        BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM
    
    if testing:
        env = lmdb.open(params.test_db_path, map_size=map_size, max_dbs=6)
    else:
        env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                if params.model_name == 'RMPI':
                    max_n_label['value'] = np.maximum(np.max(datum['en_n_labels'], axis=0), max_n_label['value'])
                else:
                    max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                    subgraph_sizes.append(datum['subgraph_size'])
                    enc_ratios.append(datum['enc_ratio'])
                    num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        if params.model_name != 'RMPI':
            txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
            txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
            txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
            txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

            txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
            txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
            txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
            txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

            txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
            txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
            txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
            txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    disclosing_subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    disclosing_subgraph = [adj[disclosing_subgraph_nodes, :][:, disclosing_subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h, enclosing_flag=True)
    disclosing_labels, disclosing_subgraph_nodes_labeled = node_label(incidence_matrix(disclosing_subgraph), max_distance=h, enclosing_flag=False)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
   
    pruned_disclosing_subgraph_nodes = np.array(disclosing_subgraph_nodes)[disclosing_subgraph_nodes_labeled].tolist()
    pruned_disclosing_labels = disclosing_labels[disclosing_subgraph_nodes_labeled]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])
        pruned_disclosing_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_disclosing_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes, pruned_disclosing_subgraph_nodes, pruned_disclosing_labels

def node_label(subgraph, max_distance=1, enclosing_flag=False):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    if enclosing_flag:
        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    else:
        # enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) < 1e6)[0]
        # process the unconnected node (neg samples)
        indices_dim0, indices_dim1 = np.where(labels == 1e7)

        indices_dim1_convert = indices_dim1 + 1
        indices_dim1_convert[indices_dim1_convert == 2] = 0
        new_indices = [indices_dim0.tolist(), indices_dim1_convert.tolist()]
        ori_indices = [indices_dim0.tolist(), indices_dim1.tolist()]

        values = labels[tuple(new_indices)] + 1
        labels[tuple(ori_indices)] = values
        # process the unconnected node (neg samples)

        # print(labels)
        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)

def bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def sp_row_vec_from_idx_list(idx_list, dim):

    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)
    
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl

def gen_meta_subgraph_datasets(args):
    data = pickle.load(open(args.pk_path, 'rb'))
    train_g = get_g(data['train_graph']['train'] + data['train_graph']['valid']
                    + data['train_graph']['test'])

    BYTES_PER_DATUM = get_average_meta_subgraph_size(args, 50, train_g) * 2
    map_size = (args.num_train_subgraph + args.num_valid_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=2)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())
    valid_subgraphs_db = env.open_db("valid_subgraphs".encode())

    for idx in tqdm(range(args.num_train_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        sup_tris, que_tris, hr2t, rt2h = sample_one_subgraph(args, train_g)
        datum = {'sup_tris':sup_tris, 'que_tris':que_tris, 'hr2t':hr2t, 'rt2h':rt2h}
        with env.begin(write=True, db=train_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))

    for idx in tqdm(range(args.num_valid_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        sup_tris, que_tris, hr2t, rt2h = sample_one_subgraph(args, train_g)
        datum = {'sup_tris':sup_tris, 'que_tris':que_tris, 'hr2t':hr2t, 'rt2h':rt2h}
        with env.begin(write=True, db=valid_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))

def sample_one_subgraph(args, bg_train_g):
    # get graph with bi-direction
    bg_train_g_undir = dgl.graph((torch.cat([bg_train_g.edges()[0], bg_train_g.edges()[1]]),
                                  torch.cat([bg_train_g.edges()[1], bg_train_g.edges()[0]])))

    # induce sub-graph by sampled nodes
    while True:
        while True:
            sel_nodes = []
            for i in range(args.rw_0):
                if i == 0:
                    cand_nodes = np.arange(bg_train_g.num_nodes())
                else:
                    cand_nodes = sel_nodes
                rw, _ = dgl.sampling.random_walk(bg_train_g_undir,
                                                 np.random.choice(cand_nodes, 1, replace=False).repeat(args.rw_1),
                                                 length=args.rw_2)
                sel_nodes.extend(np.unique(rw.reshape(-1)))
                sel_nodes = list(np.unique(sel_nodes)) if -1 not in sel_nodes else list(np.unique(sel_nodes))[1:]
            sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)

            if sub_g.num_nodes() >= 50:
                break

        sub_tri = torch.stack([sub_g.edges()[0],
                               sub_g.edata['rel'],
                               sub_g.edges()[1]])
        sub_tri = sub_tri.T.tolist()
        random.shuffle(sub_tri)

        ent_freq = ddict(int)
        rel_freq = ddict(int)
        triples_reidx = []
        ent_reidx = dict()
        entidx = 0
        for tri in sub_tri:
            h, r, t = tri
            if h not in ent_reidx.keys():
                ent_reidx[h] = entidx
                entidx += 1
            if t not in ent_reidx.keys():
                ent_reidx[t] = entidx
                entidx += 1
            ent_freq[ent_reidx[h]] += 1
            ent_freq[ent_reidx[t]] += 1
            rel_freq[r] += 1
            triples_reidx.append([ent_reidx[h], r, ent_reidx[t]])

        # randomly get query triples
        que_tris = []
        sup_tris = []
        for idx, tri in enumerate(triples_reidx):
            h, r, t = tri
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                que_tris.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                sup_tris.append(tri)

            if len(que_tris) >= int(len(triples_reidx)*0.1):
                break

        sup_tris.extend(triples_reidx[idx+1:])

        if len(que_tris) >= int(len(triples_reidx)*0.05):
            break

    # hr2t, rt2h
    hr2t, rt2h = get_hr2t_rt2h_sup_que(sup_tris, que_tris)

    return sup_tris, que_tris, hr2t, rt2h

def get_average_meta_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0
    for i in range(sample_size):
        sup_tris, que_tris, hr2t, rt2h = sample_one_subgraph(args, bg_train_g)
        datum = {'sup_tris':sup_tris, 'que_tris':que_tris, 'hr2t':hr2t, 'rt2h':rt2h}
        total_size += len(serialize(datum))
    return total_size / sample_size

def get_g(tri_list):
    triples = np.array(tri_list)
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
    g.edata['rel'] = torch.tensor(triples[:, 1].T)
    return g

def get_g_bidir(triples, args):
    g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                   torch.cat([triples[:, 2].T, triples[:, 0].T])))
    g.edata['type'] = torch.cat([triples[:, 1].T, triples[:, 1].T + args.num_rel])
    return g

def get_hr2t_rt2h(tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    return hr2t, rt2h

def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in sup_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    for tri in que_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t = tri
        que_hr2t[(h, r)] = hr2t[(h, r)]
        que_rt2h[(r, t)] = rt2h[(r, t)]

    return que_hr2t, que_rt2h

def get_indtest_test_dataset_and_train_g(args):
    data = pickle.load(open(args.pk_path, 'rb'))['ind_test_graph']
    num_ent = len(np.unique(np.array(data['train'])[:, [0, 2]]))

    hr2t, rt2h = get_hr2t_rt2h(data['train'])

    return data, num_ent, hr2t, rt2h
    # test_dataset = KGEEvalDataset(args, data['test'], num_ent, hr2t, rt2h)

    # g = get_g_bidir(torch.LongTensor(data['train']), args)

    # return test_dataset, g