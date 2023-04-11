import numpy as np
import sys
sys.path.append("..\..")

from core.node import Tensor
import core.functions as F
import torch 

def test_simple_relu():
    x=np.array([[1,2,-1],[2,3,4]])
    x_n=Tensor(x,requires_grad=True)
    res=F.relu(x_n)
    assert res.data.tolist()==[[1,2,0],[2,3,4]]

    res.backward(Tensor.ones(2,3))

    assert x_n.grad.data.tolist()==[[1,1,0],[1,1,1]]
def test_simple_leakyRelu():
    x=np.array([[1,2,-1],[2,3,4]])
    x_n=Tensor(x,requires_grad=True)
    res=F.leaky_relu(x_n)

    assert np.allclose(res.data,np.array([[1,2,-0.01],[2,3,4]]))

    res.backward(Tensor.ones(2,3))

    assert np.allclose(x_n.grad.data,np.array([[1,1,0],[1,1,1]]))

def test_simple_tanh():
    x=np.array([1,2,3,4])

    x_n=Tensor(x,requires_grad=True)

    x_t=torch.Tensor(x)

    z_n=F.tanh(x_n)
    z_t=torch.tanh(x_t)

    assert np.allclose(z_n.data,z_t.data)

test_simple_relu()
test_simple_leakyRelu()
test_simple_tanh()