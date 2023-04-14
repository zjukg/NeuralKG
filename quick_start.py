import pytorch_lightning as pl
import neuralkg_ind
from neuralkg_ind.utils.tools import *
args = neuralkg_ind.setup_parser()
data2pkl(args.dataset_name)
gen_subgraph_datasets(args)
train_sampler = import_class("SubSampler")(args)
valid_sampler = import_class("ValidSampler")(args)
test_sampler = import_class("TestSampler")(args)
kgdata = neuralkg_ind.import_class('KGDataModule') \
    (args, train_sampler, valid_sampler, test_sampler)
model = neuralkg_ind.import_class('Grail')(args)
lit_model = neuralkg_ind.import_class('indGNNLitModel') \
    (model, args)
trainer = pl.Trainer.from_argparse_args(args)
trainer.fit(lit_model, datamodule=kgdata)