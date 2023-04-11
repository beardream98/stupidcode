from decision_tree import decision_Tree
from classes import dataset,dataset_sub
import numpy as np
import pandas as pd
data=pd.read_csv("watermelondata2.csv",header=0,sep="\t",index_col="id")
import time



column_name=data.columns
y_label=column_name[-1]
x=data[column_name[:-1]]
y=pd.DataFrame(data[column_name[-1]])

def numerical_pd(data):
    column_names=data.columns
    data_num=[]
    for column_name in column_names:
        num,index=pd.factorize(data[column_name])
        # data=pd.concat([data,pd.DataFrame(num,columns={column_name+"_num"})],axis=1)
        data_num.append(pd.DataFrame(num,columns={column_name+"_num"}))
    data_num=pd.concat(data_num,axis=1)

    return data_num

x_num=numerical_pd(x)
y_num=numerical_pd(y)
x_num,y_num=np.array(x_num),np.array(y_num)
# d=dataset(x_num,y_num)
# res=d.search_unique_data(0,False)
# d0=dataset_sub(d,0,1)
# d0.show_info()
tree=decision_Tree(x_num,y_num)

tree.show_tree()

# res=tree.predict_tree([6,3])
# print(res)