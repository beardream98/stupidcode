#
# @lc app=leetcode.cn id=77 lang=python3
#
# [77] 组合
#
from typing import List
# @lc code=start
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return_list=[]
        def dfs(depth,begin,curr_list):
            if depth==k:
                return_list.append(curr_list[:])
                return 
            for i in range(begin,n):
                curr_list.append(i+1)
                dfs(depth+1,i+1,curr_list)
                curr_list.pop()
        
        dfs(0,0,[])
        return return_list
            
# @lc code=end

