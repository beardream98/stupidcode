from typing import List
from core.parameter import Parameter
from core.node import Tensor
import numpy as np
class Optimizer:
    def __init__(self, params: List[Parameter]) -> None:
        self.params = params

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    '''
    随机梯度下降
    '''

    def __init__(self, params: List[Parameter], lr: float = 1e-3) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            # p -= p.grad * self.lr
            p.assign(p-p.grad*self.lr)

class GCHECK(Optimizer):
    '''
   
    '''

    def __init__(self, params: List[Parameter], epsilon: float = 1e-3) -> None:
        super().__init__(params)
        self.epsilon = epsilon
        self.iter=self.generator()
    def generator(self):
        for param in self.params:
            for index,element in np.ndenumerate(param.data):
                param.data[index]-=self.epsilon
                yield 
                param.data[index]+=self.epsilon
    def step(self) -> None:
        next(self.iter)
        

            

if __name__=="__main__":
    pass