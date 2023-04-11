import sys

sys.path.append("..")
from core.node import debug_mode

from core.module import MSELoss,BCELoss
from core.module import Linear,LogisticRegression,normalization
from core.opt import SGD
from core.node import Tensor
from core.functions import sigmoid
import pandas as pd
import numpy as np

def load_data(path):
    data=pd.read_csv(path,header=None,names=["x1","x2","y"])
    x=data[["x1","x2"]].to_numpy()

    y=data["y"].to_numpy()
    x,y=x.reshape(-1,2),y.reshape(-1,1)
    x,y=Tensor(x),Tensor(y)    
    return x,y

path="../data/marks.txt"
x,y=load_data(path)

axis=0
ln=normalization(method="slide",axis=axis)
# x_n=ln(x)
x_n=x


model=Linear(2,1)
optimizer=SGD(model.parameters(),lr=1e-3)
loss=BCELoss("mean")
epochs=20000


for epoch in range(epochs):
    
    optimizer.zero_grad() 

    label=model(x_n)

    l=loss(label,y)
    
    optimizer.zero_grad() 
    
    l.backward()
    optimizer.step()

    if (epoch+1)%1000==0:
        total=len(y)
        correct=np.sum((label.numpy().round()==y.numpy()))
        accuracy=correct/total
        print(accuracy)
        print(f"Train -  Loss: {l.numpy().item()}. Accuracy: {accuracy}\n")