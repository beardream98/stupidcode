from collections import defaultdict
import numpy as np
class TreeNode:
    def __init__(self, attribute=None,label=None,dimension=None):
        self.attribute=attribute
        self.label = label
        self.child=[]
        self.dimension=dimension
        self.isleaf=False

class dataset:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.num_dict=defaultdict(int)
        self.unique_y=None
        self.maxlabel=None
        self.distribution()
    def distribution(self):
        for label in self.y:
            self.num_dict[label[0]]+=1
        self.unique_y=len(self.num_dict)
        max_num=0
        for key in self.num_dict.keys():
            if self.num_dict[key]>max_num:
                self.maxlabel=key
                max_num=self.num_dict[key]
    def search_unique_data(self,dimension=0,is_y=False):
        if is_y:
            return np.unique(self.y)
        else:
            # data=[ x[i][dimension] for i in range(len(x))]
            data=self.x[:,dimension]


            return np.unique(data)
            

    def show_info(self):
        print(f"num_dict:{self.num_dict}\nlen of data:{len(self.y)} \nlabel of data:{self.y}")
        print("x:",self.x)
    def __len__(self):
        return self.y.size

class dataset_sub(dataset):
    def __init__(self,D,dimension,value):
        self.x=None
        self.y=None
        self.divideByValue(D.x,D.y,dimension,value)

        super().__init__(self.x, self.y)
    def divideByValue(self,x,y,dimension,value):
        # 默认x为列表，若为矩阵等情况再改
        save_index=[]
        for i in range(len(x)):
            if x[i][dimension]==value:
                save_index.append(i)
        self.x=x[save_index]
        self.y=y[save_index]

class distribute:
    def __init__(self,dimension,values,dimension_name=None) -> None:
        self.dimension=dimension
        self.dimension_name=dimension_name
        self.values=values




if __name__=="__main__":
    pass