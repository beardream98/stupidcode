from classes import TreeNode,dataset,distribute,dataset_sub
from choose_func import Gain
from collections import deque
class decision_Tree:
    def __init__(self,x,y) -> None:
        self.x=x
        self.y=y
        self.root=TreeNode()
        self.D=dataset(self.x,self.y)
        dimensions=x.shape[1]
        self.A=self.create_A(dimensions)
        self.TreeGenerate(self.root,self.D,self.A,-1)

    def create_A(self,dimensions):
        A=[]
        for dimension in range(dimensions):
            values=self.D.search_unique_data(dimension,is_y=False)
            a=distribute(dimension,values)
            A.append(a)
        return A
        
    def is_same_value(self,D,A):
        dimensions=[ a.dimension for a in A]
        n=len(D)
        d_value=[ D.x[0,dimension] for dimension in dimensions]
        for i in range(1,n):
            new_d_value=D.x[i,dimensions]
            if not (new_d_value==d_value).all():
                return False
        return True
    def TreeGenerate(self,F_Node,D,A,node_value):
        #value作为边值
        Node=TreeNode()
        Node.value=node_value

        if len(A)==0 or self.is_same_value(D,A):
            Node.isleaf=True
            Node.label=D.maxlabel
            F_Node.child.append(Node)
            return 

        a_choose=A[0]
        max_gain=0
        for a in A:
            temp=Gain(D,a)
            if temp>max_gain:
                max_gain=temp
                a_choose=a
        new_A=[ a for a in A if a!=a_choose]

        #对节点值进行设定
        Node.dimension=a_choose.dimension
        F_Node.child.append(Node)

        for value in a_choose.values:
            d_v=dataset_sub(D,a_choose.dimension,value)
            if len(d_v)==0:
                leaf_Node=TreeNode()
                leaf_Node.value=value
                leaf_Node.isleaf=True
                leaf_Node.label=D.maxlabel
                Node.child.append(leaf_Node)
            else:
                self.TreeGenerate(Node,d_v,new_A,value)

    def predict_tree(self,x):
        Node=self.root.child[0]
        while(not Node.isleaf):
            dimension=Node.dimension
            for node in Node.child:
                if node.value==x[dimension]:
                    Node=node
                    break
        y=Node.label
        return y
    def show_tree(self):
        queue1=deque([self.root.child[0]])
        queue2=deque([])
        height=0
        print("-"*20)
        print(f"the height of Tree:{height}")
        height+=1

        while(queue1 or queue2):
            if not queue1:
                queue1=queue2
                queue2=deque([])
                print("-"*20)
                print(f"the height of Tree:{height}")
                height+=1
            Node=queue1.popleft()
            print(f"Node.val:{Node.value},Node.dimension:{Node.dimension}",end="   ")
            if not Node.child:
                print(f"y_label:{Node.label}")
                continue
            for i,child in enumerate(Node.child):
                queue2.append(child)
                print(f"child{i}.val:{child.value}",end="   ")
            print()





        
if __name__=="__main__":
    pass