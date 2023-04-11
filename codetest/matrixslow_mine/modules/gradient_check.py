import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from core.node import Tensor
from core.module import Linear,LogisticRegression,normalization

from core.module import MSELoss,BCELoss
from core.module import Linear,LogisticRegression
from core.opt import SGD,GCHECK
from core.node import Tensor
from core.functions import sigmoid

epslion=1e-4

def load_data(path):
    data=pd.read_csv(path,header=None,names=["x1","x2","y"])
    # data=data.loc[0:1]
    x=data[["x1","x2"]].to_numpy()

    y=data["y"].to_numpy()
    # x,y=x[0],y[0]
    x,y=x.reshape(-1,2),y.reshape(-1,1)
    
    

    x,y=Tensor(x),Tensor(y)    
    return x,y

path="../data/marks.txt"
x,y=load_data(path)

## 模型predict阶段
axis=0
ln=normalization(method="distrubution",axis=axis)
x_n=ln(x)
model=LogisticRegression(2,1)
loss=BCELoss("mean")


output=model(x_n)
l=loss(output,y)
l.backward()

## 模型计算的grad
grad_model=np.array([])
for p in model.parameters():
    if p.grad.ndim>1:
        p_grad=p.grad.numpy().squeeze()
    else:
        p_grad=p.grad.numpy()

    grad_model=np.concatenate([grad_model,p_grad])


##通过GCHECK来调整输入
optimizer=GCHECK(model.parameters(),epsilon=epslion)

grad_approx=np.array([])
##每次调动由GCHECK产生的生成器 会改动一个输入参数 (w1,w2,w3,...)->(w1-epslion，w2,w3) 
##用新的输入来计算输出结果，计算近似梯度 j( (w1-epslion，w2,w3)-(w1,w2,w3,...) )/epslion
## -(l1.data.item()-l.data.item()) /epslion


for _ in range(len(grad_model)):
    optimizer.step() 
    output1=model(x_n)
    l1=loss(output1,y)
    grad_one=-(l1.data.item()-l.data.item()) /epslion
    grad_approx=np.concatenate([grad_approx,np.array(grad_one).reshape(1)])


print(grad_approx,grad_model)

# 计算模型计算出的的梯度和近似梯度的欧氏距离
numerator = np.linalg.norm(grad_model - grad_approx)
denominator = np.linalg.norm(grad_model) + np.linalg.norm(grad_approx)
difference = numerator / denominator

print(difference/2)