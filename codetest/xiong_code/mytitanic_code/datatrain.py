import torch 
import torch.utils.data as Data
import torch.nn as nn
from matplotlib import pyplot as plt 
import numpy as np 
from torch.utils.data import DataLoader,Dataset 
import csv
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


class TraindataSet(Dataset):
    def __init__(self,train_features,train_labels):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

def get_kfold_data(k, i, X, y):  
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]     # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        
    return X_train, y_train, X_valid,y_valid
 
 

def k_fold(k, X_train, y_train,x_test,y_test,num_epochs=512,learning_rate=0.5, weight_decay=0, batch_size=64):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum ,valid_acc_sum = 0,0
    net=torch.nn.Sequential(
            torch.nn.Linear(7,5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5,2),
            )  ### 实例化模型
    abc=open('parameter.txt','w')

    
    for i in range(k):
        data = get_kfold_data(k, i, X_train, y_train) # 获取k折交叉验证的训练和验证数据

        para=list(net.parameters())
        print(para,file=abc)
        net.train()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,\
                                   weight_decay, batch_size) 
        para=list(net.parameters())
        print(para,file=abc)
        
        print('*'*25,'第',i+1,'折','*'*25)
        print('train_loss:%.6f'%train_ls[-1][0],'train_acc:%.4f\n'%train_ls[-1][1],\
              'valid loss:%.6f'%valid_ls[-1][0],'valid_acc:%.4f'%valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
    abc.close()

    print('#'*10,'最终k折交叉验证结果','#'*10) 
    print('train_loss_sum:%.4f'%(train_loss_sum/k),'train_acc_sum:%.4f\n'%(train_acc_sum/k),\
          'valid_loss_sum:%.4f'%(valid_loss_sum/k),'valid_acc_sum:%.4f'%(valid_acc_sum/k))



    print('#'*10,'最终测试集结果','#'*10)
    test_ls=log_rmsetest(1,net,x_test, y_test)
    print('test_loss:%.6f'%(test_ls[0]),'test_acc:%.4f'%(test_ls[1]))

#########训练函数##########
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate,weight_decay, batch_size):
    train_ls, test_ls = [], [] ##存储train_loss,test_loss
    dataset = TraindataSet(train_features, train_labels) 
    train_iter = DataLoader(dataset, batch_size, shuffle=True) 
    ### 将数据封装成 Dataloder 对应步骤（2）

    #这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr= learning_rate, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:  ###分批训练 
            output  = net(X)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### 每128 个epoch 记录一次
        if (epoch+1) %128==0:
            train_ls.append(log_rmse(0,net, train_features, train_labels)) 
            test_ls.append(log_rmse(1,net, test_features, test_labels))
    return train_ls, test_ls
 
def log_rmse(flag,net,x,y):
    if flag == 1: ### valid 数据集
        net.eval() #在测试时使用 关闭batch 和dropout
    output = net(x)
    result = torch.max(output,1)[1].view(y.size())
    corrects = (result.data == y.data).sum().item()
    accuracy = corrects*100.0/len(y)  #### 5 是 batch_size
    loss = loss_func(output,y)
    net.train()
    return (loss.data.item(),accuracy)
def log_rmsetest(flag,net,x,y):
    if flag == 1: ### valid 数据集
        net.eval() #在测试时使用 关闭batch 和dropout
    output = net(x)
    result = torch.max(output,1)[1].view(y.size())
    corrects = (result.data == y.data).sum().item()
    accuracy = corrects*100.0/len(y)  #### 5 是 batch_size
    loss = loss_func(output,y)
    net.train()

    output_ydata=result.numpy()

    filename='gender_submission.csv'
    my_sub=open('my_submission.csv','w',newline="")
    csv_write = csv.writer(my_sub,dialect='excel')
    i=0
    with open(filename) as f:
        reader=csv.reader(f)
        header_row=next(reader)
        csv_write.writerow(header_row)
        for reader_column in reader:
            reader_column[1]=output_ydata[i]
            i=i+1
            csv_write.writerow(reader_column)
    my_sub.close()

    return (loss.data.item(),accuracy)

    
loss_func = nn.CrossEntropyLoss() ###申明loss函
k_fold(5,x,y,x_test,y_test) ### k=10,十折交叉验证