import importlib
from IPython import embed
import os
import time
import yaml

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
