import inspect
import math
from typing import List
import numpy as np
import sys

if __name__=="__main__":
    sys.path.append("..")


from core.node import Tensor
from core.parameter import Parameter

import core.functions as F

class Module:
    '''
    所有模型的基类
    '''
    def __init__(self) -> None:
        self._paramerters=self.parameters()


    def parameters(self) -> List[Parameter]:
        parameters = []
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                parameters.append(value)
            elif isinstance(value, Module):
                parameters.extend(value._paramerters)

        return parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

class Linear(Module):
    r"""
         对给定的输入进行线性变换: :math:`y=xA^T + b`

        Args:
            in_features: 每个输入样本的大小
            out_features: 每个输出样本的大小
            bias: 是否含有偏置，默认 ``True``
        Shape:
            - Input: `(*, H_in)` 其中 `*` 表示任意维度，包括none,这里 `H_{in} = in_features`
            - Output: :math:`(*, H_out)` 除了最后一个维度外，所有维度的形状都与输入相同，这里H_out = out_features`
        Attributes:
            weight: 可学习的权重，形状为 `(out_features, in_features)`.
            bias:   可学习的偏置，形状 `(out_features)`.
        """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None
        self.reset_parameters()
        super().__init__()
        
    def reset_parameters(self) -> None:
        self.weight.assign(np.random.randn(self.out_features, self.in_features))

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs @ self.weight.T
        if self.bias is not None:
            x = x + self.bias

        return x

class LogisticRegression(Module):
    def __init__(self, input_dim, output_dim):
        self.linear = Linear(input_dim, output_dim)

        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(self.linear(x))

class SoftmaxRegression(Module):
    def __init__(self, input_dim, output_dim):
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.linear(x))
        

###########################################################
"""
normalization
"""
class normalization(Module):
    def __init__(self,method:str="slide",axis:int=0):

        """ normalization方法
        method slide:平移 distrubution :正态
        
        """
        self.axis=axis
        self.method=method
    
    def forward(self,x:Tensor)->Tensor:

        if self.method=="slide":
            range_x=x.max_attr(self.axis)-x.min_attr(self.axis)
            new_x=(x-x.min_attr(self.axis))/range_x

        elif self.method=="distrubution":
            new_x=(x-x.mean_attr(self.axis))/(x.var_attr(self.axis))
        
        return new_x



#####################################################
#loss

class Loss(Module):
    '''
    损失的基类
    '''
    reduction: str  # none | mean | sum

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

class MSELoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        '''
        均方误差
        '''
        super().__init__(reduction)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        assert inputs.size == target.size
        
        errors = (inputs - target) ** 2
        if self.reduction == "mean":
            loss = errors.sum(keepdims=False) / len(inputs)
        elif self.reduction == "sum":
            loss = errors.sum(keepdims=False)
        else:
            loss = errors

        return loss

class BCELoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        
        super().__init__(reduction)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return F.binary_entropy(inputs,target,self.reduction)

class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)


    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        
        return F.cross_entropy(inputs,target,self.reduction)
