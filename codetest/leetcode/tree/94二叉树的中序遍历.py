# Definition for a binary tree node.
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
