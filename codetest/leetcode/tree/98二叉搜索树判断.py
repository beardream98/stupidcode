# Definition for a binary tree node.
from tkinter.tix import Tree


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


def isValidBST(root):
    mid_val=float("-inf")
    stack_tree=[]
    top=root
    while(top or len(stack_tree)!=0):
        while(top):
            stack_tree.append(top)
            top=top.left
        top=stack_tree.pop()
        if top.val>=mid_val:
            mid_val=top.val
        else:
            return False
        top=top.right
    return True
    


tree_list=eval(input("enter a tree list:"))
root=makeTree(tree_list)
print(isValidBST(root))

            

        