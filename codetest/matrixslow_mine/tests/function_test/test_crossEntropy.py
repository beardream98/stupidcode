import numpy as np
import sys
sys.path.append("..\..")

from core.node import Tensor
from core.functions import softmax
from core.functions import cross_entropy,sigmoid,binary_entropy,bad_cross_entropy

import torch
import torch.nn.functional as F
def test_simple_binaryEntropy():
    x=Tensor([3])
    y=Tensor([1])
    
    tx = torch.tensor(np.array([3]), dtype=torch.float32, requires_grad=True)
    ty = torch.tensor(np.array([1]), dtype=torch.float32)

    mo = torch.binary_cross_entropy_with_logits(tx, ty).mean()

    error_old=-y*sigmoid(x).log()-(1-y)*(1-sigmoid(x)).log()
    error_new=binary_entropy(x,y)

    np.testing.assert_allclose(error_new.data,mo.data)
    


def test_simple_crossEntropy():
    x=Tensor([0.1,0.5,0.4])
    y=Tensor([1,0,0])

    error=bad_cross_entropy(x,y,"sum")
    np.testing.assert_almost_equal(error.data,(-1*Tensor(0.1).log()).data)

def test_simlpe_stable_ce():
    x=Tensor([10,20,30])
    y=Tensor([1,0,0])

    #与bad版本结果一致
    x1=softmax(x)
    error1=bad_cross_entropy(x1,y,"sum")
    error2=cross_entropy(x,y,"sum")
    np.testing.assert_almost_equal(error1.data,error2.data)

def test_stable_cs():
    x=np.array([[1000,100,1]])
    y=np.array([[1,0,0]])

    x_n,y_n=Tensor(x),Tensor(y)

    x_t,y_t=torch.Tensor(x),torch.Tensor(np.array([0])).long()
    torch_loss = torch.nn.NLLLoss()
    error_t = torch_loss(torch.log_softmax(x_t, dim=-1, dtype=torch.float32), y_t)

    error_n=cross_entropy(x_n,y_n)

    
    assert error_t.data.item()==error_n.data.item()

test_simple_binaryEntropy()  
test_simple_crossEntropy()
test_simlpe_stable_ce()

test_stable_cs()
