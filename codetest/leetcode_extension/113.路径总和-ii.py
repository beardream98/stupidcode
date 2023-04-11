#
# @lc app=leetcode.cn id=113 lang=python3
#
# [113] 路径总和 II
#
from typing import Optional,List

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        return_list=[]

        def dfs(Node,left_val,curr_list):
            if not Node.left and not Node.right:
                if left_val==Node.val:
                    return_list.append(curr_list[:]+[Node.val])
                return 
            left_val-=Node.val
            curr_list.append(Node.val)
            if Node.left:
                dfs(Node.left,left_val,curr_list)
            if Node.right:
                dfs(Node.right,left_val,curr_list)
            left_val+=Node.val

            curr_list.pop()
        if root:
            dfs(root,targetSum,[])
            return return_list

        else:
            return []

# @lc code=end

