# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import pytorch_lightning as pl
from neuralkg_ind import lit_model
from neuralkg_ind import data
def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument('--model_name', default="TransE", type=str, help='The name of model.')
    parser.add_argument('--dataset_name', default="FB15K237", type=str, help='The name of dataset.')
    parser.add_argument('--data_class', default="KGDataModule", type=str, help='The name of data preprocessing module, default KGDataModule.')
    parser.add_argument("--litmodel_name", default="KGELitModel", type=str, help='The name of processing module of training, evaluation and testing, default KGELitModel.')
    parser.add_argument("--train_sampler_class",default="UniSampler",type=str, help='Sampling method used in training, default UniSampler.')
    parser.add_argument("--valid_sampler_class",default=None, type=str, help='Sampling method used in validation, default TestSampler.')
    parser.add_argument("--test_sampler_class",default="TestSampler",type=str, help='Sampling method used in validation and testing, default TestSampler.')
    parser.add_argument('--loss_name', default="Adv_Loss", type=str, help='The name of loss function.')
    parser.add_argument('--negative_adversarial_sampling','-adv', default=True, action='store_false', help='Use self-adversarial negative sampling.')
    parser.add_argument('--optim_name', default="Adam", type=str, help='The name of optimizer')
    parser.add_argument("--seed", default=321, type=int, help='Random seed.')
    parser.add_argument('--margin', default=12.0, type=float, help='The fixed margin in loss function. ')
    parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
    parser.add_argument('--emb_dim', default=200, type=int, help='The embedding dimension in KGE model.')
    parser.add_argument('--out_dim', default=200, type=int, help='The output embedding dimmension in some KGE model.')
    parser.add_argument('--num_neg', default=10, type=int, help='The number of negative samples corresponding to each positive sample')
    parser.add_argument('--num_ent', default=None, type=int, help='The number of entity, autogenerate.')
    parser.add_argument('--num_rel', default=None, type=int, help='The number of relation, autogenerate.')
    parser.add_argument('--check_per_epoch', default=5, type=int, help='Evaluation per n epoch of training.')
    parser.add_argument('--check_per_step', default=0, type=int, help='Evaluation per n step of training.')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='If the number of consecutive bad results is n, early stop.')
    parser.add_argument("--num_layers", default=2, type=int, help='The number of layers in some GNN model.')
    parser.add_argument('--regularization', '-r', default=0.0, type=float)
    parser.add_argument("--decoder_model", default=None, type=str, help='The name of decoder model, in some model.')
    parser.add_argument('--eval_task', default="link_prediction", type=str, choices=['link_prediction', 'triple_classification'], help='The task of validation and test')
    parser.add_argument("--calc_hits", default=[1,3,10], type=lambda s: [int(item) for item in s.split(',')], help='calc hits list')
    parser.add_argument('--filter_flag', default=True, action='store_false', help='Filter in negative sampling.')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='Select the GPU in training, default cuda:0.')
    parser.add_argument("--use_wandb", default=False, action='store_true',help='Use "weight and bias" to record the result.')
    parser.add_argument('--use_weight', default=False, action='store_true', help='Use subsampling weight.')
    parser.add_argument('--checkpoint_dir', default="", type=str, help='The checkpoint model path')
    parser.add_argument('--save_config', default=False, action='store_true', help='Save paramters config file.')
    parser.add_argument('--load_config', default=False, action='store_true', help='Load parametes config file.')
    parser.add_argument('--config_path', default="", type=str, help='The config file path.')
    parser.add_argument('--freq_init', default=4, type=int)
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--shuffle', default=True, action='store_false')
    parser.add_argument('--norm_flag', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout Layer')  #TODO graip 0 CrossE 0.5

    #parser only for Ruge
    parser.add_argument('--slackness_penalty', default=0.01, type=float)

    #parser only for CompGCN
    parser.add_argument("--opn", default='corr',type=str, help="only on CompGCN, choose Composition Operation")
    
    #parser only for BoxE
    parser.add_argument("--dis_order", default=2, type=int, help="only on BoxE, the distance order of score")
    
    # parser only for ComplEx_NNE
    parser.add_argument('--mu', default=10, type=float, help='only on ComplEx_NNE,penalty coefficient for ComplEx_NNE')
    
    # paerser only for KBAT
    parser.add_argument('--epoch_GAT', default=3000, type=int, help='only on KBAT, the epoch of GAT model')
    parser.add_argument("-p2hop", "--partial_2hop", default=False, action='store_true')

    # parser only for CrossE
    parser.add_argument('--neg_weight', default=50, type=int, help='only on CrossE, make up label')
    
    # parer only for ConvE
    parser.add_argument('--emb_shape', default=20, type=int, help='only on ConvE,The first dimension of the reshaped 2D embedding')
    parser.add_argument('--inp_drop', default=0.2, type=float, help='only on ConvE,Dropout for the input embeddings')
    parser.add_argument('--hid_drop', default=0.3, type=float, help='only on ConvE,Dropout for the hidden layer')
    parser.add_argument('--fet_drop', default=0.2, type=float, help='only on ConvE,Dropout for the convolutional features')
    parser.add_argument('--hid_size', default=9728, type=int, help='only on ConvE,The side of the hidden layer. The required size changes with the size of the embeddings.')
    parser.add_argument('--hid_size_component', default=3648, type=int, help='only on ConvE,The side of the hidden layer. The required size changes with the size of the embeddings.')
    parser.add_argument('--smoothing', default=0.1, type=float, help='only on ConvE,Make the label smooth')
    parser.add_argument("--out_channel", default=32, type=int, help="only on ConvE")    
    parser.add_argument("--ker_sz", default=3, type=int, help="only on ConvE")
    parser.add_argument("--k_h", default=10, type=int, help="only on ConvE")
    parser.add_argument("--k_w", default=20, type=int, help="only on ConvE")
    parser.add_argument("--fc_bias", default=True, action='store_false', help="only on ConvE, the bias of fc in ConvE layer")

    #parser only for SEGNN #TODO: short parser
    parser.add_argument("--kg_layer", default=1, type=int, help="only on SEGNN")
    parser.add_argument("--rm_rate", default=0.5, type=float, help= "only on SEGNN")
    parser.add_argument("--ent_drop", default=0.2, type=float, help="only on SEGNN")
    parser.add_argument("--rel_drop", default=0, type=float, help="only on SEGNN")
    parser.add_argument("--ent_drop_pred", default=0.3, type=float, help="only on ConvE")
    parser.add_argument("--fc_drop", default = 0.1, type=float, help = "only on SEGNN")
    parser.add_argument("--comp_op", default='mul', type=str, help="only on SEGNN")
    parser.add_argument("--bn", default=False, action='store_true')
    parser.add_argument("--warmup_epoch", default=5, type=int, help="only on SEGNN")
    parser.add_argument("--warm_up_steps", default=None, type=int, help="only on SEGNN")
    parser.add_argument("--maxsteps", default=None, type=int, help="only on SEGNN")
    parser.add_argument("--pred_rel_w", default=False, action="store_true", help="only on SEGNN")   
    parser.add_argument("--label_smooth", default=0.1, type=float, help="only on SEGNN")
 
    # parser only for IterE
    parser.add_argument("--max_entialments", default=2000, type=int, help="in IterE.py")
    parser.add_argument("--axiom_types", default=10, type=int, help="in IterE.py")
    parser.add_argument("--select_probability", default=0.8, type=float, help="in IterE.py")
    parser.add_argument("--axiom_weight", default=1.0, type=float, help="in IterE.py")
    parser.add_argument("--inject_triple_percent", default=1.0, type=float, help="in IterE.py")
    parser.add_argument("--update_axiom_per",default=2, type=int, help='in IterELitModel.py')

    #parser only for HAKE
    parser.add_argument("--phase_weight", default=1.0, type=float, help='only on HAKE,The weight of phase part')
    parser.add_argument("--modulus_weight", default=1.0, type=float, help='only on HAKE,The weight of modulus part')

    #parser only for DualE
    parser.add_argument("--regularization_two", default=0, type=float, help='only on DualE, regularization_two')
    
    #parser for loggging
    parser.add_argument("--save_path", type=str, default='logging')
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--special_name",type=str, default=None)

    #parser for indGNN model
    parser.add_argument("--inductive", default=False, action='store_true', help='using the inductive inference setting')
    parser.add_argument("--inp_dim", type=int, default=8)  # NOTE: set the n_feat_dim 
    parser.add_argument("--aug_num_rels", type=int, default=None)
    parser.add_argument("--max_n_label", type=int, default=None)
    parser.add_argument("--db_path", type=str, default=None, help='specify the path for subgraph db')
    parser.add_argument("--pk_path", type=str, default=None, help='specify the path for pickle file')
    parser.add_argument("--test_db_path", type=str, default=None, help='specify the path for test subgraph db')
    parser.add_argument("--l2", type=float, default=5e-4, help="Regularization constant for GNN weights")
    parser.add_argument("--reduction", type=str, choices=['mean', 'sum', 'none'], default='mean', help="specify the reduction to apply to the output")
    #model
    parser.add_argument("--num_bases", "-b", type=int, default=100, help="Number of basis functions to use for GCN weights")
    parser.add_argument("--kge_model", type=str, default="TransE", help="Which KGE model to load entity embeddings from")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False, help='whether to use pretrained KGE embeddings')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True, help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum', help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True, help='whether to have attn in model or not')
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32, help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32, help="Relation embedding size for attention")
    parser.add_argument("--edge_dropout", type=float, default=0.5, help="Dropout rate in edges of the subgraphs")
    #sampler
    parser.add_argument("--hop", type=int, default=3, help="Enclosing subgraph hop number")
    parser.add_argument("--enclosing_sub_graph", "-en", type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument("--num_neg_samples_per_link", "-neg", type=int, default=1, help="Number of negative examples to sample per positive link")
    #presubgraph
    parser.add_argument("--max_links", type=int, default=1000000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--constrained_neg_prob", "-cn", type=float, default=0.0, help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--directed", type=bool, default=False, help='Whether subgraph is directed or undirected')
    #SNRI
    parser.add_argument('--sort_data', type=bool, default=True, help='whether to training data according to relation id ')
    parser.add_argument('--init_nei_rels', type=str, choices=['no', 'out', 'in', 'both'], default='in', help='the manner of utilizing relatioins when initializing entity embedding')
    parser.add_argument('--sem_dim', type=int, default=24, help='the dimension of sematic part of node embedding')
    parser.add_argument('--max_nei_rels', type=int, default=10, help='the maximum num of neighbor relations of each node when initialzing the node embedding.')
    parser.add_argument('--nei_rels_dropout', type=float, default=0.4, help='Dropout rate in aggregating relation embeddings.')
    parser.add_argument('--is_comp', type=str, default='mult', choices=['mult', 'sub'], help='The composition manner of node and relation')
    parser.add_argument('--comp_ht', type=str, choices=['mult', 'mlp', 'sum'], default='sum', help='The composition operator of head and tail embedding')
    parser.add_argument('--coef_dgi_loss', type=float, default=5, help='Coefficient of MI loss')
    parser.add_argument('--nei_rel_path', action='store_false', help='whether to consider neighboring relational paths')
    parser.add_argument('--path_agg', type=str, choices=['mean', 'att'], default='att', help='the manner of aggreating neighboring relational paths.')
    #RMPI
    parser.add_argument('--target2nei_atten', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--conc', action='store_true', help='apply target-aware attention for 2-hop neighbors')
    parser.add_argument('--ablation', type=int, default=0, help='0,1 correspond to base, NE')
    #MorsE
    parser.add_argument('--num_train_subgraph', type=int, default=10000, help='the number of train subgraph')
    parser.add_argument('--num_valid_subgraph', type=int, default=200, help='the number of valid subgraph')
    parser.add_argument('--rw_0', default=10, type=int, help='the times of random walk')
    parser.add_argument('--rw_1', default=10, type=int, help='the number of paths')
    parser.add_argument('--rw_2', default=5, type=int, help='the length of random walk')
    # Get data, model, and LitModel specific arguments
    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)

    data_group = parser.add_argument_group("Data Args")
    data.BaseDataModule.add_to_argparse(data_group)


    parser.add_argument("--help", "-h", action="help")
    
    
    return parser
