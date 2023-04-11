from core.node import Tensor

## activation function 

def sigmoid(x:Tensor) -> Tensor:
    # if x<0:
    #     return x.exp() / ( 1 + x.exp() )
    # else:
    return 1/(1+(-x).exp())

def bad_softmax(x:Tensor) -> Tensor:
    y=x.exp()
    return y/y.sum()

def softmax(x:Tensor) -> Tensor:
    y=(x-x.max()).exp()

    return y/y.sum()

def relu(x:Tensor)->Tensor:
    return x.clip(floor=0)

def leaky_relu(x:Tensor,a=0.01)->Tensor:
    return x.clip(floor=(a*x).data,cell=None)

def tanh(x:Tensor)->Tensor:

    return (x.exp()-(-x).exp()) / ( x.exp()+(-x).exp() )


def binary_entropy(input:Tensor,label:Tensor,reduction:str="mean"):
    """
    input :(N,1)
    label :one-hot vector (N,1) or (1,)
    N: size of sample
    default activation : sigmoid 
    """
    N=len(label)

    # max(0,z)+ zy +log(1+e^(-|z|))

    error=input.clip(floor=0,cell=None) - input*label + (1 + (-abs(input)) .exp() ).log()
    if reduction=="mean":
        return error.sum()/N
    elif reduction=="sum":
        return error.sum()
    else:
        return error


def bad_cross_entropy(input:Tensor,label:Tensor,reduction:str="mean"):
    """
    input :(N,K)
    label :one-hot vector (N,K) or (K,)
    N: size of sample
    K: the number of label type 
    default activation : None
    """
    N=len(label)

    assert input.shape[-1] == label.shape[-1]
    error=( -label*input.log() )
    if reduction=="mean":
        return error.sum()/N
    elif reduction=="sum":
        return error.sum()
    else:
        return error

def logsumexp(x:Tensor,axis=-1):
    x_max=x.max(axis=axis,keepdims=True)

    y=x-x_max
    # b+log(sum(e^(x-b) ))
    return x_max+( (y.exp()).sum(axis=axis,keepdims=True) ).log()


def cross_entropy(input:Tensor,label:Tensor,reduction:str="mean"):
    """
    input :(N,K)
    label :one-hot vector (N,K) or (K,)
    N: size of sample
    K: the number of label type 
    default activation : softmax
    """
    N=len(label)

    assert input.shape[-1] == label.shape[-1]
    # log( sum(e^z) ) - sum(y*z)
    error= logsumexp(input) - ( input*label).sum(axis=-1,keepdims=True)

    if reduction=="mean":
        return error.sum()/N
    elif reduction=="sum":
        return error.sum()
    else:
        return error

if __name__=="__main__":
    pass