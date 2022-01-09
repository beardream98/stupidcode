import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# bn的模板
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03

input_size=[1,10,5,1]
# 不含最后一层的激活
ACTIVATION =[torch.relu,torch.relu,torch.relu]
N_HIDDEN = len(input_size) # 算最后一层预测

class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)   # 1代表输入的维度

        for i in range(N_HIDDEN-1):               # build hidden layers and BN layers
            fc = nn.Linear(input_size[i], input_size[i+1])
            setattr(self, 'fc%i' % i, fc)       # 将层导入模型中
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(input_size[i+1], momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
                self.bns.append(bn)
        self.predict = nn.Linear(input_size[N_HIDDEN-2], input_size[N_HIDDEN-1])         # output layer
        self._set_init(self.predict)            # parameters initialization

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)     # input batch normalization
        layer_input = [x]
        for i in range(N_HIDDEN-1):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION[i](x)
            layer_input.append(x)
        out = self.predict(x)
        return out, layer_input, pre_activation
net=Net(batch_normalization=True)
print(net)