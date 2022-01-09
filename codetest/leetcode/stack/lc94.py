# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
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


a=TreeNode(1)
b=TreeNode(2)
c=TreeNode(3)
a.right=b
b.left=c
print(inorderTraversal(a))

        
        

