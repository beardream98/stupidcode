
import sys
sys.path.append("../..")
from core.node import Tensor
import numpy as np

def test_simple_sum():
    x=Tensor([1,2,3],requires_grad=True)
    y=x.sum()
    assert y.data.item()==6
    
    y.backward()

    assert x.grad.data.tolist()==[1,1,1]

def test_matrix_sum():
    x=Tensor.ones(*(3,3),**{"requires_grad":True})
    y=x.sum()

    assert y.data.item()==9

    y.backward()

    assert x.grad.data.tolist()==np.ones_like(x.data).tolist()



def test_matrix_with_axis():
    x=Tensor.ones(*(3,3),**{"requires_grad":True})
    y=x.sum(axis=0)

    assert y.shape==(3,)

    y.backward([1,1,1])

    assert x.grad.data.tolist()==np.ones_like(x.data).tolist()

def test_matrix_with_keepdims():
    
    x=Tensor.ones(*(3,3),**{"requires_grad":True})
    y=x.sum(axis=0,keepdims=True)

    assert y.shape==(1,3)

    y.backward([1,1,1])

    assert x.grad.data.tolist()==np.ones_like(x.data).tolist()

test_simple_sum()
test_matrix_sum()
test_matrix_with_axis()
test_matrix_with_keepdims()