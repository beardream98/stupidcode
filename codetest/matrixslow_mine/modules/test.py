import sys
sys.path.append("..")
import numpy as np
from core.module import MSELoss,BCELoss
from core.module import Linear,LogisticRegression,normalization
from core.opt import SGD,GCHECK
from core.node import Tensor
from core.functions import sigmoid
import pandas as pd



def load_data(path):
    data=pd.read_csv(path,header=None,names=["x1","x2","y"])
    x=data[["x1","x2"]].to_numpy()

    y=data["y"].to_numpy()
    x,y=x.reshape(-1,2),y.reshape(-1,1)
    
    

    x,y=Tensor(x),Tensor(y)    
    return x,y

path="../data/marks.txt"

x,y=load_data(path)
data=pd.read_csv(path,header=None,names=["x1","x2","y"])
x=data[["x1","x2"]].to_numpy()

y=data["y"].to_numpy()
x,y=x.reshape(-1,2),y.reshape(-1,1)
x,y=Tensor(x),Tensor(y)

axis=0
ln=normalization(method="distrubution",axis=axis)
x_n=ln(x)



model=LogisticRegression(2,1)
optimizer=GCHECK(model.parameters(),epsilon=1e-7)
loss=BCELoss("mean")

output=model(x_n)
l=loss(output,y)
l.backward()
def step(params):
    for param in params:
        for index,element in np.ndenumerate(param.data):
            param.data[index] -= 1e-7
            yield param.data[index]
            param.data[index] += 1e-7



its=optimizer.step()
for it in its:
    print(it)
    print(model.parameters())
    