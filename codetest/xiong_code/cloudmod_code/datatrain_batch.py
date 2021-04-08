import torch 
import torch.utils.data as Data
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
    torch.nn.Linear(200,50),
    torch.nn.Tanh(),
    torch.nn.Linear(50,1),
)

#  batch size 部分
Batch_Size=64
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=Batch_Size,
    shuffle=False,
    num_workers=3
)

def show_test():
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
    for t in range(100):
        for step,(batch_x,batch_y) in enumerate (loader):
            prediction = net(batch_x)     # 喂给 net 训练数据 x, 输出预测值

            loss=loss_func(prediction, batch_y)     # 计算两者的误差

            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        if t%10==0:
            prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
            loss1=loss_func(prediction, y)     # 计算两者的误差
            print("epoch=:"+str(t)+" loss: ",end="")
            print(loss1)

    prediction=net(x_test)
    loss=loss_func(prediction, y_test)
    print("loss in test data: ",end="")
    print(loss)
if __name__ == '__main__':
    show_test()
