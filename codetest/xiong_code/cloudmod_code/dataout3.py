#coding:utf-8
import numpy as np  
import matplotlib.pyplot as plt 
import math
def plot_cloud_model(Ex, En, He,n):  
    X = np.random.normal(loc=En, scale=He, size=n)  
    for i in range(n):  
        Enn = X[i]  
        X[i] = np.random.normal(loc=Ex, scale=np.abs(Enn), size=1)
    
    return  X
# 设置 Ex En 样本维数
ex1,ex2=0,0
en1,en2=1,2
He1,He2=0.1,0.5
n=200

# 样本数量 同时控制he 不超过 en的三分之一
data_num=2000

# 数据初始化
x_data=np.zeros((data_num,n))
y_data=np.zeros((data_num))

for i in range(0,1000):
    num=plot_cloud_model(ex1, en1, He1, n) 
    x_data[i,:]=num
    y_data[i]=1
for i in range(1000,data_num):
    num=plot_cloud_model(ex2, en2, He2, n) 
    x_data[i,:]=num
    y_data[i]=0
# 打乱数据排序
shuffle_ix = np.random.permutation(np.arange(len(x_data)))
x_data = x_data[shuffle_ix]
y_data = y_data[shuffle_ix]

# 拆分 数据集和验证集 9：1
train_max=math.floor(data_num*9/10)
x_train=x_data[0:train_max,:]
x_test=x_data[train_max:data_num,:]
y_train=y_data[0:train_max]
y_test=y_data[train_max:data_num]

# 保存
np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)
