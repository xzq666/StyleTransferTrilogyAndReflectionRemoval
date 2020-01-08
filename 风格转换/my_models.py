# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2020-01-08 09:16

import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs
