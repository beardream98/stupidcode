from core.node import Tensor
from typing import Any,Tuple
import numpy as np


class Function:
    def __init__(self,*tensors:"Tensor") -> None:
        """
        *输入存储为元组 **输入存储为字典
        为了防止将非data作为节点输入，需要使用变量名=xx的赋值方式
        """
        self.depends_on=[ t for t in tensors]
        self.save_tensors=[]
    def __new__(cls, *args, **kwargs):
        '''__new__是静态方法，当该类被实例化时调用'''
        # 把这两个方法转换为静态方法，我们可以通过类名直接调用 cls实例
        cls.forward = staticmethod(cls.forward)
        cls.backward = staticmethod(cls.backward)
        cls.apply = staticmethod(cls.apply)
        return super().__new__(cls)
    def save_for_backward(ctx, *x: Any) -> None:
        ctx.save_tensors.extend(x)

    def forward(ctx, *args: Any, **kwargs: Any) -> np.ndarray:
        '''前向传播，进行真正运算的地方'''
        raise NotImplementedError("You must implement the forward function for custom Function.")

    def backward(ctx, grad: Any) -> Any:
        '''实现反向传播，计算梯度'''
        raise NotImplementedError("You must implement the backward method for your custom Function "
                                  "to use it with backward mode AD.")

    def apply(self, *xs: "Tensor", **kwargs) -> "Tensor":
        '''与PyTorch一样，我们也不直接调用forward，而是调用此方法'''
    
        ctx = self(*xs) 

        ret = Tensor(self.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))
        
        if ret.requires_grad:
            ret._ctx = ctx

        return ret

def unbroadcast(grad: np.ndarray, in_shape: Tuple) -> np.ndarray:
    '''
    广播操作的逆操作，确保grad转换成in_shape的形状
    Args:
        grad: 梯度
        in_shape: 梯度要转换的形状
    Returns:
    '''
    # 首先计算维度个数之差
    ndims_added = grad.ndim - len(in_shape)
    # 由于广播时，先从左边插入，再进行复制，所以逆操作时，也从左边开始，进行复制的逆操作（求和）
    for _ in range(ndims_added):
        # 在axis=0上进行求和，去掉第0个维度，如果ndims_added > 1，就需要不停的在第0个维度上面求和
        grad = grad.sum(axis=0)

    for i,dim in enumerate(in_shape):
        if dim==1:
            grad=grad.sum(axis=i,keepdims=True)

    return grad

class Add(Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        实现 z = x + y ，我们这里的x和y都是Numpy数组，因此可能发生广播，
        在实现反向传播是需要注意
        '''
        return x+y

    def backward(ctx, grad: Any) -> Any:
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        inshape1,inshape2=[t.shape for t in ctx.depends_on]
        return unbroadcast(grad,inshape1),unbroadcast(grad,inshape2)

class Sub(Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        return x-y

    def backward(ctx, grad: Any) -> Any:
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        inshape1,inshape2=[t.shape for t in ctx.depends_on]
        return unbroadcast(grad,inshape1),unbroadcast(-grad,inshape2)


class Mul(Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        实现 z = x * y
        '''
        # 乘法需要保存输入x和y，用于反向传播
        ctx.save_for_backward(x,y)
        return x*y

    def backward(ctx, grad: Any) -> Any:
        x, y = ctx.save_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y
        inshape1,inshape2=[t.shape for t in ctx.depends_on]
        return unbroadcast(grad*y,inshape1),unbroadcast(grad*x,inshape2)

class TrueDiv(Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        实现 z = x / y
        '''
        ctx.save_for_backward(x, y)
        return x / y

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.save_tensors
        return unbroadcast(grad / y, x.shape), unbroadcast(grad * (-x / y ** 2), y.shape)

class Matmul(Function):
    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        z = x @ y
        '''
        assert x.ndim>1 and y.ndim>1
        
        ctx.save_for_backward(x,y)

        return x @ y

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x,y=ctx.save_tensors
        return unbroadcast(grad@y.swapaxes(-2,-1),x.shape),unbroadcast(x.swapaxes(-2,-1)@grad,y.shape)

class Sum(Function):
    def forward(ctx,x:np.ndarray,axis=None,keepdims=False)->np.ndarray:
        return x.sum(axis,keepdims=keepdims)
    
    def backward(ctx, grad: Any) -> Any:
        x,=ctx.depends_on
        
        return np.broadcast_to(grad,x.shape)

class Neg(Function):
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        return -grad

epslion=0
class Log(Function):
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        
        ctx.save_for_backward(x)
        return np.log(x+epslion)

    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        
        x,=ctx.save_tensors
        return grad/(x+epslion)

class Exp(Function):
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x)
        return np.exp(x)

    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        x, = ctx.save_tensors
        return grad*np.exp(x)    

class Pow(Function):
    def forward(ctx, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, c)
        return x ** c

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, None]:
        x, c = ctx.save_tensors
        # 把c当成一个常量，不需要计算梯度
        return grad * c * x ** (c - 1), None

class Reshape(Function):
    def forward(ctx, x: np.ndarray, shape: Tuple) -> np.ndarray:
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(ctx, grad: np.ndarray) -> Tuple[np.ndarray, None]:
        x_shape, = ctx.save_tensors
        return grad.reshape(x_shape), None


class Max(Function):
    def forward(ctx, x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        ret = np.amax(x, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(x,axis,ret,keepdims)
        return ret


    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        x, axis, ret, keepdims = ctx.save_tensors
        mask=(x==ret)
        div=mask.sum(axis,keepdims=keepdims)
        return grad*mask/div

class Clip(Function):
    def forward(ctx, x: np.ndarray, floor:None ,cell=None) -> np.ndarray:
    
        if not isinstance(floor ,(np.ndarray,float,int,type(None))) or \
            not isinstance(cell ,(np.ndarray,float,int,type(None))) :
            print("please make sure the type of cell or floor in (int,float,np.ndarray)")
            assert False 

        if not isinstance(floor,np.ndarray) and floor==None:
            floor=x.min()
        if not isinstance(cell,np.ndarray) and cell==None:
            cell=x.max()
        
        ctx.save_for_backward(x,floor,cell)
        return np.clip(x,floor,cell)


    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        x,floor,cell=ctx.save_tensors
        mask=(x>=floor) * (x<=cell)
        return grad*mask

class Abs(Function):
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x)
        return np.abs(x)

    def backward(ctx, grad: np.ndarray) -> np.ndarray:
        x,=ctx.save_tensors
        mask=np.ones(x.shape)*(x>0) + -1*np.ones(x.shape)*(x<0)

        return grad*mask


class getitem(Function):
    def forward(ctx, x: np.ndarray, idxs: Any) -> np.ndarray:
        '''
        z = x[idxs]
        '''
        # 如果传入[1:3]，变成切片slice
        # 如果idxs传入单个索引，会被看成是整数，所以这里转换回来
        if isinstance(idxs,np.ndarray) and idxs.shape==():
            idxs=int(idxs.item())

        ctx.save_for_backward(x.shape, idxs)
        return x[idxs]

    def backward(ctx, grad) -> Tuple[np.ndarray, None]:

        x_shape, idxs = ctx.save_tensors
        bigger_grad = np.zeros(x_shape, dtype=grad.dtype)
        bigger_grad[idxs] = grad

        return bigger_grad, None

class Transpose(Function):
    def forward(ctx, x: np.ndarray, axes) -> np.ndarray:
        ctx.save_for_backward(axes)
        return x.transpose(axes)

    def backward(ctx, grad: np.ndarray) -> Any:
        axes, = ctx.save_tensors
        if axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(axes))), None

if __name__=="__main__":
    pass