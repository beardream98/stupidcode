import torch 
import torch.utils.data as Data
import torch.nn as nn
from matplotlib import pyplot as plt 
import numpy as np 
x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

x=torch.from_numpy(x_train)
y=torch.from_numpy(y_train)

x=x.float()
y=y.long()

x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
x_test=x_test.float()
y_test=y_test.long()

net=torch.nn.Sequential(
    torch.nn.Linear(200,50),
    torch.nn.ReLU(),
    torch.nn.Linear(50,20),
    torch.nn.ReLU(),
    torch.nn.Linear(20,5),
    torch.nn.Sigmoid(),
    torch.nn.Linear(5,2)
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()      # 预测值和真实值的误差计算公式 (均方差)

t_plot=[]
accuracy_plot=[]
fig, ax = plt.subplots() # 创建图实例

for t in range(1001):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%200==0:
        plt.cla()
        prediction=torch.max(out,1)[1]
        pred_y=prediction.numpy()
        target_y=y.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

        t_plot.append(t)
        accuracy_plot.append(accuracy)
        print("epoch=:"+str(t)+" accuracy: ",end="")
        print(accuracy)
plt.plot(t_plot,accuracy_plot,c='r')
ax.plot(t_plot, accuracy_plot, label='train_set') # 作y1 = x 图，并标记此线名为linear

out=net(x_test)
loss=loss_func(out,y_test)
prediction=torch.max(out,1)[1]
pred_y=prediction.numpy()
target_y=y_test.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("test accurancy:",accuracy)
test_accurancy=[accuracy]*len(t_plot)
ax.plot(t_plot, test_accurancy, label='test_set') #作y2 = x^2 图，并标记此线名为quadratic

plt.xlabel('Steps')
ax.set_xlabel('Steps') #设置x轴名称 x label
ax.set_ylabel('accuracy') #设置y轴名称 y label
ax.set_title('Classification Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示
plt.show()