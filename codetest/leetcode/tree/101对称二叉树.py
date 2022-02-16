# Definition for a binary tree node.
import queue


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSymmetric(root):
    left,right=root.left,root.right

    queue_left=[left]
    queue_right=[right]

    while(len(queue_left)!=0 and len(queue_right)!=0):
        left=queue_left.pop(0)
        right=queue_right.pop(0)
        if left==None and right==None:
            continue
        elif left==None or right==None:
            return False
        if left.val!=right.val:
            return False
        queue_left.append(left.left)
        queue_left.append(left.right)
        queue_right.append(right.right)
        queue_right.append(right.left)
    return True
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

# tree_list=eval(input("enter a tree list:"))
# root=makeTree(tree_list)
# print(isSymmetric(root))
