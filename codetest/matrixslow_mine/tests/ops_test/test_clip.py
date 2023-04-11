import sys
sys.path.append("../..")
from core.node import Tensor
import numpy as np

def test_simple_clip():
    x=Tensor([1,2,3,4,5],requires_grad=True)
    floor,cell=2,4
    z=x.clip(floor=floor,cell=cell)
    assert z.data.tolist()==[2,2,3,4,4]

    z.backward(Tensor([1,1,1,1,1]))
    assert x.grad.data.tolist()==[0,1,1,1,0]
def test_array_clip():
    # test for floor :array 
    x=np.arange(1,10).reshape(3,3)
    x=Tensor(x,requires_grad=True)

    floor=2*np.ones((3,3))
    cell=4*np.ones((3,3))
    
    z=x.clip(floor=floor,cell=cell)
    assert z.data.min()>=2 and z.data.max()<=4

    z.backward(Tensor.ones())
    
    assert x.grad.data.tolist()==[[0,1,1],[1,0,0],[0,0,0]]

test_simple_clip()

test_array_clip()