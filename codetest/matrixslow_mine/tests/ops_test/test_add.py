import sys
sys.path.append("../..")
from core.node import Tensor
import numpy as np

def test_simple_add():
    x=Tensor(1,requires_grad=True)
    y=1

    assert (x+y).data.item()==2
    
    z=x+y
    z.backward()
    
    assert x.grad.data.item() == 1 

def test_array_add():
    x=Tensor([1,2,3],requires_grad=True)
    y=Tensor([1,2,3],requires_grad=True)
    z=x+y

    assert z.data.tolist()==[2,4,6]


    z.backward(Tensor([1,1,1]))
    assert x.grad.data.tolist()==[1,1,1]
    
    x+=1
    assert x.grad==None
def test_broadcast_add():
    x=Tensor.randn(*(2,3),**{"requires_grad":True})
    y=Tensor.randn(*(1,3),**{"requires_grad":True})

    z=x+y
    assert z.shape==(2,3)

    z.backward(Tensor.ones(2,3))
    assert y.grad.data.tolist()==[[2,2,2]]


test_simple_add()
test_array_add()
test_broadcast_add()