from core.node import Tensor
from typing import Any,Tuple,Union
import numpy as np

Arrayable=Union[float,list,np.ndarray]


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor]) -> None:
        
        if isinstance(data, Tensor):
            data = data.data
        # Parameter都是需要计算梯度的
        super().__init__(data, requires_grad=True)


if __name__=="__main__":
    pass