import numpy as np
import sys
sys.path.append("..\..")

from core.node import Tensor

from core.functions import softmax,bad_softmax
def test_simple_softmax():
    x=Tensor([1,2,3])
    y=bad_softmax(x)
    assert y.sum().data.item()==1

    
test_simple_softmax()