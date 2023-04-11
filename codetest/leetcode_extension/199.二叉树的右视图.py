#
# @lc app=leetcode.cn id=199 lang=python3
#
# [199] 二叉树的右视图
#
from inspect import stack
from typing import List
from collections import deque
# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        return_list=[]
        queue1,queue2=deque([root]),deque()
        return_list=[root.val]
        while(queue1):
            head=queue1.popleft()
            if head.left:queue2.append(head.left)
            if head.right:queue2.append(head.right)
            if not queue1 and  queue2:
                queue1=queue2.copy()
                queue2=deque()
                return_list.append(queue1[-1].val)
                    
        return return_list
# @lc code=end
from classes import TreeNode
from functions import makeTree,inorderTraversal




