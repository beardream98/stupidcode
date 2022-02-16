# Definition for a binary tree node.
from turtle import right
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
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
    

def lowestCommonAncestor(root, p,q):
    Ancestor_Node=None
    max_height=-1
    def isancestor(root,height):
        nonlocal max_height,Ancestor_Node
        if root==None:
            return 0
        left_query=isancestor(root.left,height+1)
        right_query=isancestor(root.right,height+1)
        root_query= 1 if root.val==p or root.val==q else 0

        if left_query+right_query+root_query>=2:
            if height>max_height:
                Ancestor_Node=root
                max_height=height
        return max(left_query,right_query,root_query)
    isancestor(root,0)
    return Ancestor_Node
tree_list=eval(input("please enter a tree list:"))
root=makeTree(tree_list)
ancestor_node=lowestCommonAncestor(root,5,1)
print(ancestor_node.val)
# [3,5,1,6,2,0,8,"null","null",7,4]

        