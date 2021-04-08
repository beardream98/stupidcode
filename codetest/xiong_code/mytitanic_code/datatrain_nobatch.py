import torch 
import torch.utils.data as Data
import torch.nn as nn
from matplotlib import pyplot as plt 
import numpy as np 
from torch.utils.data import DataLoader,Dataset 
import csv
from torch import nn
from torch.nn import init

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
input_size=[7,20,15,10,5,2]
# 不含最后一层的激活
ACTIVATION =[torch.relu,torch.relu,torch.relu,torch.relu]
N_HIDDEN = len(input_size) # 算最后一层预测

class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(input_size[0], momentum=0.5)   

        for i in range(N_HIDDEN-2):               
            fc = nn.Linear(input_size[i], input_size[i+1])
            setattr(self, 'fc%i' % i, fc)       # 将层导入模型中
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(input_size[i+1], momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
                self.bns.append(bn)
        self.predict = nn.Linear(input_size[N_HIDDEN-2], input_size[N_HIDDEN-1])        
        self._set_init(self.predict)            # parameters initialization

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0)
        
    def forward(self, x):
        if self.do_bn: 
            x = self.bn_input(x)     # input batch normalization
        for i in range(N_HIDDEN-2):
            try:
                x = self.fcs[i](x)
            except Exception:
                print(i)
                print(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION[i](x)
        out = self.predict(x)
        return out
net=Net(batch_normalization=True)

optimizer = torch.optim.SGD(params=net.parameters(), lr= 0.1)
loss_func = torch.nn.CrossEntropyLoss() 
for epoch in range(2000):
    output  = net(x)
    loss = loss_func(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 200 == 0:
        prediction = torch.max(output, 1)[1]  #第一个是value 我们不需要 我们要的只是value 所在的索引 代表着标签
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print("epoch:",end="")
        print(epoch,end=" ")
        print("accuracy:",accuracy)

output  = net(x_test)
prediction = torch.max(output, 1)[1]  
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

print("accuracy:",accuracy)

filename='gender_submission.csv'
my_sub=open('my_submission.csv','w',newline="")
csv_write = csv.writer(my_sub,dialect='excel')
i=0
with open(filename) as f:
        reader=csv.reader(f)
        header_row=next(reader)
        csv_write.writerow(header_row)
        for reader_column in reader:
                reader_column[1]=pred_y[i]
                i=i+1
                csv_write.writerow(reader_column)

my_sub.close()