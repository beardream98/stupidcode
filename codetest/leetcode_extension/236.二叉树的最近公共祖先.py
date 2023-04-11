#
# @lc app=leetcode.cn id=236 lang=python3
#
# [236] 二叉树的最近公共祖先
#
from classes import TreeNode

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        return_node=None
        def dfs(root):
            nonlocal return_node
            if not root:
                return 0,0
            #回溯
            p_in=1 if root==p else 0
            q_in=1 if root==q else 0

            p_in_son,q_in_son=dfs(root.left)
            p_in,q_in=max(p_in,p_in_son),max(q_in,q_in_son)

            p_in_son,q_in_son=dfs(root.right)
            p_in,q_in=max(p_in,p_in_son),max(q_in,q_in_son)

            if p_in and q_in and return_node==None :
                return_node=root
            return p_in,q_in
        dfs(root)
        return return_node
# @lc code=end


root,p,q=None,None,None
return_node=None
def dfs(root):
    if not root:
        return 0,0
    #回溯
    p_in=1 if root==p else 0
    q_in=1 if root==q else 0

    p_in_son,q_in_son=dfs(root.left)
    p_in,q_in=max(p_in,p_in_son),max(q_in,q_in_son)

    p_in_son,q_in_son=dfs(root.right)
    p_in,q_in=max(p_in,p_in_son),max(q_in,q_in_son)

    if p_in and q_in and return_node==None :
        return_node=root
    return p_in,q_in


