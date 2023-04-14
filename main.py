# -*- coding: utf-8 -*-
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from neuralkg_ind.utils import setup_parser
from neuralkg_ind.utils.tools import *
from neuralkg_ind.data.Sampler import *
from neuralkg_ind.data.Grounding import GroundAllRules


def main():
    parser = setup_parser() #设置参数
    args = parser.parse_args()
    if args.load_config:
        args = load_config(args, args.config_path)
    seed_everything(args.seed)

    if args.inductive:
        if not os.path.exists(args.pk_path):
            data2pkl(args.dataset_name)

        if args.model_name == 'MorsE':
            if not os.path.exists(args.db_path):
                gen_meta_subgraph_datasets(args)
        else:
            if not os.path.exists(args.db_path): 
                gen_subgraph_datasets(args) # [头， 尾， 关系]
    
    if args.init_checkpoint:
        override_config(args) #TODO: set checkpoint autoloading
    elif args.data_path is None :
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args=args)

    logging.info("++++++++++++++++++++++++++loading hyper parameter++++++++++++++++++++++++++")
    for key, value in args.__dict__.items():
        logging.info("Parameter "+key+":  "+str(value))
    logging.info("++++++++++++++++++++++++++++++++over loading+++++++++++++++++++++++++++++++")
    
    """set up sampler to datapreprocess""" #设置数据处理的采样过程
    train_sampler_class = import_class(f"neuralkg_ind.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
    
    test_sampler_class = import_class(f"neuralkg_ind.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的

    if args.valid_sampler_class != None:
        valid_sampler_class = import_class(f"neuralkg_ind.data.{args.valid_sampler_class}")
        valid_sampler = valid_sampler_class(train_sampler)
    else:
        valid_sampler = test_sampler

    """set up datamodule""" #设置数据模块
    data_class = import_class(f"neuralkg_ind.data.{args.data_class}") #定义数据类 DataClass
    kgdata = data_class(args, train_sampler, valid_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"neuralkg_ind.model.{args.model_name}")
    
    if args.model_name == "RugE":
        ground = GroundAllRules(args)
        ground.PropositionalizeRule()

    if args.model_name == "ComplEx_NNE_AER":
        model = model_class(args, train_sampler.rel2id)
        
    elif args.model_name == "IterE":
        print(f"data.{args.train_sampler_class}")
        model = model_class(args, train_sampler, test_sampler)
    else:
        model = model_class(args)

    if args.model_name == 'SEGNN':
        src_list = train_sampler.get_train_1.src_list
        dst_list = train_sampler.get_train_1.dst_list
        rel_list = train_sampler.get_train_1.rel_list

    """set up lit_model"""
    litmodel_class = import_class(f"neuralkg_ind.lit_model.{args.litmodel_name}")

    if args.model_name =='SEGNN':
        lit_model = litmodel_class(model, args, src_list, dst_list, rel_list)
    else:
        lit_model = litmodel_class(model, args)
    
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="NeuralKG_ind")
        logger.log_hyperparams(vars(args))
    if args.inductive and args.model_name != 'MorsE':
        """early stopping"""
        early_callback = pl.callbacks.EarlyStopping(
            monitor="Eval|auc",
            mode="max",
            patience=args.early_stop_patience,
            # verbose=True,
            check_on_train_epoch_end=False,
        )
        """set up model save method"""
        dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval|auc",
            mode="max",
            filename="{epoch}-{Eval|auc:.3f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        callbacks = [early_callback, model_checkpoint]
    else:
        """early stopping"""
        early_callback = pl.callbacks.EarlyStopping(
            monitor="Eval|mrr",
            mode="max",
            patience=args.early_stop_patience,
            # verbose=True,
            check_on_train_epoch_end=False,
        )
        """set up model save method"""
        dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval|mrr",
            mode="max",
            filename="{epoch}-{Eval|mrr:.3f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        callbacks = [early_callback, model_checkpoint]
    # initialize trainer
    if args.model_name == "IterE":

        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus="0,",
            check_val_every_n_epoch=args.check_per_epoch, 
            reload_dataloaders_every_n_epochs=1 # IterE
        )
    elif args.check_per_step:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus="0,",
            val_check_interval=args.check_per_step,
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus="0,",
            check_val_every_n_epoch=args.check_per_epoch,
        )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)
    if args.use_wandb:
        logger.watch(lit_model)     
    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # 加载本次实验中dev上表现最好的模型，进行test
        path = model_checkpoint.best_model_path
    else:
        path = args.checkpoint_dir
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)

if __name__ == "__main__":
    main()
