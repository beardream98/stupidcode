#coding:utf-8
import numpy as np  
import matplotlib.pyplot as plt 
import math
def plot_cloud_model(Ex, En, He,n):  
    X = np.random.normal(loc=En, scale=He, size=n)  
    for i in range(n):  
        Enn = X[i]  
        X[i] = np.random.normal(loc=Ex, scale=np.abs(Enn), size=1)
    
    return  np.square(X)
# 设置 Ex En 样本维数
ex=0
en=1
n=1000

# 样本数量 同时控制he 不超过 en的三分之一
data_num=10000
he_step=en/(data_num*3)

# 数据初始化
x_data=np.zeros((data_num,n))
y_data=np.zeros((data_num))

for i in range(data_num):
    num=plot_cloud_model(0, 1, he_step*i, n) 
    x_data[i,:]=num
    y_data[i]=he_step*i

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


    
