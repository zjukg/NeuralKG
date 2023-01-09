# -*- coding: utf-8 -*-
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from IPython import embed
import wandb
from neuralkg.utils import setup_parser
from neuralkg.utils.tools import *
from neuralkg.data.Sampler import *
from neuralkg.data.Grounding import GroundAllRules


def main():
    parser = setup_parser() #设置参数
    args = parser.parse_args()
    if args.load_config:
        args = load_config(args, args.config_path)
    seed_everything(args.seed) 
    """set up sampler to datapreprocess""" #设置数据处理的采样过程
    train_sampler_class = import_class(f"neuralkg.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
    #print(train_sampler)
    test_sampler_class = import_class(f"neuralkg.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的
    """set up datamodule""" #设置数据模块
    data_class = import_class(f"neuralkg.data.{args.data_class}") #定义数据类 DataClass
    kgdata = data_class(args, train_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"neuralkg.model.{args.model_name}")
    
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
    litmodel_class = import_class(f"neuralkg.lit_model.{args.litmodel_name}")
    
    if args.model_name =='SEGNN':
        lit_model = litmodel_class(model, args, src_list, dst_list, rel_list)
    else:
        lit_model = litmodel_class(model, args)
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="NeuralKG")
        logger.log_hyperparams(vars(args))
    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval|mrr",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )
    """set up model save method"""
    # 目前是保存在验证集上mrr结果最好的模型
    # 模型保存的路径
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
