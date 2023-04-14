# -*- coding: utf-8 -*-
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from neuralkg_ind.utils import setup_parser
from neuralkg_ind.utils.tools import *
from neuralkg_ind.data.Sampler import *
from tqdm import tqdm

def main(arg_path):

    args = setup_parser()
    args = load_config(args, arg_path)
    seed_everything(args.seed) 

    print('This demo is powered by \033[1;32mNeuralKG_ind \033[0m')

    train_path = input("\033[1;32mPlease input the train graph path: \033[0m") #./dataset/NELL_v1/
    args.data_path = train_path

    support_entity = set()
    support_relation = set()
    print('Loading the train dataset...')    
    with open(train_path+'train.txt','r') as f:
        for line in tqdm(f.readlines()):
            h, r, t = line.strip().split()
            support_entity.add(h)
            support_relation.add(r)
            support_entity.add(t)
    support_relation = list(support_relation)
    print(f'\033[1;32mThe support relation set\033[0m: [{support_relation[0]}, {support_relation[1]}, {support_relation[2]}, ...]')
    print(f'\033[1;32mAttention:\033[0m The relation of query should exist in the relation set.')

    model_path = input("\033[1;32mPlease input the model path: \033[0m") # /Grail/demo.ckpt
    model_path = './config'+model_path
    args.checkpoint_dir = model_path

    test_path = input("\033[1;32mPlease input the test support graph path: \033[0m")  # ./dataset/NELL_v1_ind/
    test_triple = dict()
    with open(test_path+'test.txt', 'r') as f:
        for (id, line) in enumerate(f.readlines()):
            h, r, t = line.strip().split()
            test_triple[(h,r)] = (t, id)
    print('Use the trained model to predict tail entity, in the inductive task setting. Please input query.')

    while 1:
        query = input("\033[1;32mQuery: \033[0m") # (televisionstation_wvta, branch_office, ?) or (televisionstation_wvta, agent_belongs_to_organization, ?)
        head = 'concept:'+query[1:-1].split(', ')[0].replace('_',':')
        relation = 'concept:'+query[1:-1].split(', ')[1].replace('_','')
        if relation in support_relation:
            print(f"\033[1;32mRelation is in the train dataset. \033[0m Start reasoning stage.")
            ground_truth = '_'.join(test_triple[(head, relation)][0].split(':')[1:])
            idx = test_triple[(head, relation)][1]
            break
        else:
            print(f"\033[1;31mRelation is not in the train dataset. \033[0m Please input another query." )

    if args.inductive:

        if not os.path.exists(args.pk_path):
            data2pkl(args.dataset_name)

        if not os.path.exists(args.db_path): 
            gen_subgraph_datasets(args) # [头， 尾， 关系]

    """set up sampler to datapreprocess""" #设置数据处理的采样过程
    train_sampler_class = import_class(f"neuralkg_ind.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的

    valid_sampler_class = import_class(f"neuralkg_ind.data.{args.valid_sampler_class}")
    valid_sampler = valid_sampler_class(train_sampler)

    test_sampler_class = import_class(f"neuralkg_ind.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的

    """set up datamodule""" #设置数据模块
    data_class = import_class(f"neuralkg_ind.data.{args.data_class}") #定义数据类 DataClass
    kgdata = data_class(args, train_sampler, valid_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"neuralkg_ind.model.{args.model_name}")
    model = model_class(args)
    
    """set up lit_model"""
    litmodel_class = import_class(f"neuralkg_ind.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args)
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")

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

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus='0,',
        check_val_every_n_epoch=args.check_per_epoch,
    )
    '''保存参数到config'''

    if not args.test_only:
        trainer.fit(lit_model, datamodule=kgdata)
        test_path = model_checkpoint.best_model_path
    else:
        test_path = args.checkpoint_dir

    trainer.test(lit_model, datamodule=kgdata)

    lit_model.load_state_dict(torch.load(test_path)["state_dict"])
    lit_model.eval()

    (with_neg_data, target) = train_sampler.test_triples[idx]['head']
    data = test_sampler.get_subgraphs(with_neg_data, train_sampler.adj_list, \
        train_sampler.dgl_adj_list, train_sampler.args.max_n_label,{},{})
    score = lit_model.model.cpu()(data).squeeze(1).detach()
    _, index = score.topk(10)
    index = with_neg_data[index][:, 1]
    top10_ent = ['_'.join(train_sampler.id2ent[i].split(':')[1:]) for i in index]
    rank = np.argwhere(np.argsort(score.numpy())[::-1] == target) + 1   

    print(f"\033[1;32mTop 10 Prediction:\033[0m{top10_ent}")
    print(f"\033[1;32mGroud Truth: \033[0m{ground_truth}   \033[1;32mRank: \033[0m{rank[0][0]}")
    print('This demo is powered by \033[1;32mNeuralKG_ind \033[0m')

if __name__ == "__main__":
    main(arg_path = 'config/Grail_demo_kg.yaml')
