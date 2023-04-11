import sys
sys.path.append("../..")
from core.node import Tensor
import numpy as np

def test_simple_abs():
    x=Tensor([-4,5,0],requires_grad=True)
    z=abs(x)
    
    assert z.data.tolist()==[4,5,0]
    
    z.backward(Tensor([1,1,1]))
    
    assert x.grad.data.tolist()==[-1,1,0]

test_simple_abs()
    