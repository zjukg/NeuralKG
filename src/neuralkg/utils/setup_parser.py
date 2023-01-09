# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import pytorch_lightning as pl
from neuralkg import lit_model
from neuralkg import data
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
    parser.add_argument('--early_stop_patience', default=5, type=int, help='If the number of consecutive bad results is n, early stop.')
    parser.add_argument("--num_layers", default=2, type=int, help='The number of layers in some GNN model.')
    parser.add_argument('--regularization', '-r', default=0.0, type=float)
    parser.add_argument("--decoder_model", default=None, type=str, help='The name of decoder model, in some model.')
    parser.add_argument('--eval_task', default="link_prediction", type=str, help='The task of validation, default link_prediction.')
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
    parser.add_argument('--dropout', default=0.5, type=float, help='only on CrossE,for Dropout')
    parser.add_argument('--neg_weight', default=50, type=int, help='only on CrossE, make up label')
    
    # parer only for ConvE
    parser.add_argument('--emb_shape', default=20, type=int, help='only on ConvE,The first dimension of the reshaped 2D embedding')
    parser.add_argument('--inp_drop', default=0.2, type=float, help='only on ConvE,Dropout for the input embeddings')
    parser.add_argument('--hid_drop', default=0.3, type=float, help='only on ConvE,Dropout for the hidden layer')
    parser.add_argument('--fet_drop', default=0.2, type=float, help='only on ConvE,Dropout for the convolutional features')
    parser.add_argument('--hid_size_component', default=3648, type=int, help='only on ConvE,The side of the hidden layer. The required size changes with the size of the embeddings.')
    parser.add_argument('--hid_size', default=9728, type=int, help='only on ConvE,The side of the hidden layer. The required size changes with the size of the embeddings.')
    parser.add_argument('--smoothing', default=0.1, type=float, help='only on ConvE,Make the label smooth')
    parser.add_argument("--out_channel", default=32, type=int, help="in ConvE.py")    
    parser.add_argument("--ker_sz", default=3, type=int, help="in ConvE.py")
    parser.add_argument("--ent_drop_pred", default=0.3, type=float, help="in ConvE.py")
    parser.add_argument("--k_h", default=10, type=int, help="in ConvE.py")
    parser.add_argument("--k_w", default=20, type=int, help="in ConvE.py")

    #parser only for SEGNN
    parser.add_argument("--kg_layer", default=1, type=int, help="in SEGNN.py")
    parser.add_argument("--rm_rate", default=0.5, type=float, help= "in SEGNN.py")
    parser.add_argument("--ent_drop", default=0.2, type=float, help="in SEGNN.py")
    parser.add_argument("--rel_drop", default=0, type=float, help="in SEGNN.py")
    parser.add_argument("--fc_drop", default = 0.1, type=float, help = "in SEGNN.py")
    parser.add_argument("--comp_op", default='mul', type=str, help="in SEGNN.py")
    #WN18RR
    #parser.add_argument("--bn", default=True, action='store_true')
    #FB15K237 
    parser.add_argument("--bn", default=False, action='store_true')

    parser.add_argument("--warmup_epoch", default=5, type=int, help="in SEGNN.py")
    parser.add_argument("--warm_up_steps", default=None, type=int, help="in SEGNN.py")
    parser.add_argument("--maxsteps", default=None, type=int, help="in SEGNN.py")
    #WN18RR
    #parser.add_argument("--pred_rel_w", default=True, action="store_true")
    #FB15K237
    parser.add_argument("--pred_rel_w", default=False, action="store_true")   
    parser.add_argument("--label_smooth", default=0.1, type=float, help="in SEGNN.py")
        
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
    
    # Get data, model, and LitModel specific arguments
    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)

    data_group = parser.add_argument_group("Data Args")
    data.BaseDataModule.add_to_argparse(data_group)


    parser.add_argument("--help", "-h", action="help")
    
    
    return parser
