from typing import ValuesView


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def makeTree(tree_list):
    if len(tree_list)==0:
        return None
    Node_list=[]
    for i,tree_val in enumerate(tree_list):
        if tree_val=="null":
            Node_list.append(None)
            continue
        val=int(tree_val)
        Node=TreeNode(val=val)
        Node_list.append(Node)
        if i%2==1 and (i-1)/2>=0:
            Node_list[(i-1)//2].left=Node
        elif i%2==0 and (i-2)/2>=0:
            Node_list[(i-2)//2].right=Node
    return Node_list[0]
def inorderTraversal(root):
    stack_root=[]
    mid_visit=[]
    top=root

    #top 作为当前节点 在退栈时能告诉栈顶节点左子树已经访问完毕
    while len(stack_root)!=0 or top!=None :
        #top != None 是在右节点进栈之前可能存在的栈空
        while(top!=None):
            stack_root.append(top)
            top=top.left
        top=stack_root[-1]
        stack_root.pop()
        mid_visit.append(top.val)
        top=top.right
    return mid_visit           
def flatten(root):
    stack_root=[]
    view_Node=[]
    Visit_val=[]
    while len(stack_root)!=0 or root!=None:
        while(root!=None):
            stack_root.append(root)
            view_Node.append(root)
            Visit_val.append(root.val)
            root=root.left
        root=stack_root[-1]
        stack_root.pop()
        root=root.right
    for i in range(len(view_Node)):
        view_Node[i].left=None
        if i!=len(view_Node)-1:
            view_Node[i].right=view_Node[i+1]
        else:
            view_Node[i].right=None
    return Visit_val
     
tree_list=eval(input("enter tree list:"))
root=makeTree(tree_list)
print(inorderTraversal(root))
print(flatten(root))
        

        