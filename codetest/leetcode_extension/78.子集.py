#
# @lc app=leetcode.cn id=78 lang=python3
#
# [78] 子集
#
from typing import List

# @lc code=start
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        return_list=[]
        n=len(nums)
        def dfs(path,add_empty,begin):
            if add_empty:
                return_list.append(path[:])
                return 

            for i in range(begin,n):
                path.append(nums[i])
                dfs(path,False,i+1)
                path.pop()
            dfs(path,True,begin)
        dfs([],False,0)
        return return_list
# @lc code=end

