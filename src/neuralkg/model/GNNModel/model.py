import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
    
    def init_emb(self):
        raise NotImplementedError

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.args.num_layers):
            layer_idx = self.build_hidden_layer(idx)
            self.layers.append(layer_idx)

    def build_hidden_layer(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError