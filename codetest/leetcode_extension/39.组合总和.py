#
# @lc app=leetcode.cn id=39 lang=python3
#
# [39] 组合总和
#
from secrets import choice
from tkinter import N
from typing import List
# @lc code=start

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        n=len(candidates)
        return_list=[]
        def dfs(begin,left_value,curr_list):
            if left_value==0:
                
                return_list.append(curr_list[:])
                return 
            elif left_value<0:
                return
            
            for i in range(begin,n):
                curr_list.append(candidates[i])
                left_value-=candidates[i]
                #取i代表可重复选值
                dfs(i,left_value,curr_list)
                left_value+=candidates[i]
                curr_list.pop()
        dfs(0,target,[])
        return return_list
# @lc code=end
