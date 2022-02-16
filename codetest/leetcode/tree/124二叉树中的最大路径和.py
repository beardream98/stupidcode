# Definition for a binary tree node.
from audioop import maxpp
from logging.config import valid_ident


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxPathSum(root):
    max_pathsum=0
    def maxgain(root):
        if root==None:
            return 0
        left_gain=maxgain(root.left)
        right_gain=maxgain(root.right)
        temp_pathsum=root.val
        if left_gain>0:
            temp_pathsum+=left_gain
        if right_gain>0:
            temp_pathsum+=right_gain
        if temp_pathsum>max_pathsum:
            max_pathsum=temp_pathsum
        return max(left_gain,right_gain)+root.val if max(left_gain,right_gain)>0 else root.val

