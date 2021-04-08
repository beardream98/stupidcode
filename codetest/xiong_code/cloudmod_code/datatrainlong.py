import torch 
import torch.utils.data as Data
import torch.nn as nn
from matplotlib import pyplot as plt 
import numpy as np   
x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')
# 矩阵化
x=torch.from_numpy(x_train)
y=torch.from_numpy(y_train)
y=torch.unsqueeze(y,dim=1)
x=x.float()
y=y.float()

x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
y_test=torch.unsqueeze(y_test,dim=1)
x_test=x_test.float()
y_test=y_test.float()

net=torch.nn.Sequential(    
    torch.nn.Linear(10000,1600),
    torch.nn.ReLU(),
    torch.nn.Linear(1600,1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000,600),
    torch.nn.ReLU(),
    torch.nn.Linear(600,200),
    torch.nn.ReLU(),
    torch.nn.Linear(200,50),
    torch.nn.ReLU(),
    torch.nn.Linear(50,1)
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1,momentum=0.9)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

for t in range(2001):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    loss=loss_func(prediction, y)     # 计算两者的误差
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    if t%200==0:
        print("epoch=:"+str(t)+" loss: ",end="")
        print(loss)


prediction=net(x_test)
loss=loss_func(prediction, y_test)
print("loss in test data: ",end="")
print(loss)
py1=torch.cat((prediction,y_test),1)
print(py1)