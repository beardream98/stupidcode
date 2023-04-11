from posixpath import split
import sys

sys.path.append("..")
from core.node import debug_mode

from core.module import MSELoss,BCELoss,CrossEntropyLoss
from core.module import Linear,normalization,SoftmaxRegression
from core.opt import SGD
from core.node import Tensor
from core.functions import sigmoid
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data():
    iris=datasets.load_iris()

    x,y=iris["data"],iris["target"]
    y=np.eye(3)[y]
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=10)
    return Tensor(x_train),Tensor(x_test),Tensor(y_train),Tensor(y_test)
if __name__=="__main__":
    x_train,x_test,y_train,y_test=load_data()

    model=Linear(4,3)
    optimizer=SGD(model.parameters(),lr=1e-2)
    loss=CrossEntropyLoss("mean")
    epochs=20000


    for epoch in range(epochs):
    
        optimizer.zero_grad() 
        output=model(x_train)
        l=loss(output,y_train)
        l.backward()
        optimizer.step()
        if (epoch+1)%100==0:
            total=len(y_train)
            correct=np.sum(np.argmax(output.data,axis=1)==np.argmax(y_train.data,axis=1))
            accuracy=correct/total
            print(f"Train -  Loss: {l.numpy().item()}. Accuracy: {accuracy}\n")

    output=model(x_test)
    total=len(y_test)
    correct=np.sum(np.argmax(output.data,axis=1)==np.argmax(y_test.data,axis=1))
    accuracy=correct/total
    print(f"Accuracy: {accuracy}\n")

