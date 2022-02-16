from inspect import stack
from itertools import accumulate
from turtle import st


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def inorderTraversal(root):
    mid_visit=[]
    stack_tree=[]
    top=root
    while(top or len(stack_tree)!=0):
        while(top):
            stack_tree.append(top)
            top=top.left
        top=stack_tree.pop()
        mid_visit.append(top.val)
        top=top.right
        
    return mid_visit
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
#递归方法
class Solution:
    def path_re(self,root):
        if root==None:
            return [],[]
        pa=[root.val]
        ps=[]
        pa_l,ps_l=self.path_re(root.left)
        pa_r,ps_r=self.path_re(root.right)
        pa+=[i+root.val for i in pa_l]
        pa+=[i+root.val for i in pa_r]
        ps=pa_l+ps_l+pa_r+ps_r
        return pa,ps
        
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        pa,ps=self.path_re(root)
        cnt=0
        for i in pa:
            if i==targetSum:
                cnt+=1
        for i in ps:
            if i==targetSum:
                cnt+=1
        return cnt

#遍历所有路径保存后变为连续数组找和问题

mp={0:1}
cnt=0
acc_num=0
def dfs(root):
    global acc_num,cnt,mp
    if root==None:
        return 
    acc_num+=root.val
    if acc_num-target_sum in mp.keys():
        cnt+=mp[acc_num-target_sum]
    if acc_num not in mp.keys():
        mp[acc_num]=1
    else:
        mp[acc_num]+=1
    if root.left or root.right:
        dfs(root.left)
        dfs(root.right)
    mp[acc_num]-=1
    acc_num-=root.val 

# [10,5,-3,3,2,"null",11,3,-2,"null",1]
tree_list=eval(input("please enter a tree list:"))
target_sum=eval(input("please enter a number:"))
root=makeTree(tree_list)
# print(inorderTraversal(root))
dfs(root)
print(cnt)





    

        
