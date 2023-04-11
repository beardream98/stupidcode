#
# @lc app=leetcode.cn id=47 lang=python3
#
# [47] 全排列 II
#
from typing import List
# @lc code=start
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        visited=[0]*n
        return_list=[]
        def dfs(index,curr_list):
            if index==n:
                return_list.append(curr_list[:])
            exist_dict={}
            for i in range(n):
                if visited[i]==0 and exist_dict.get(nums[i],-1)==-1:

                    exist_dict[nums[i]]=1

                    curr_list.append(nums[i])
                    visited[i]=1
                    dfs(index+1,curr_list)
                    visited[i]=0
                    curr_list.pop()
        dfs(0,[])
        return return_list
# @lc code=end

