#
# @lc app=leetcode.cn id=491 lang=python3
#
# [491] 递增子序列
#
from typing import List

# @lc code=start
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        return_list=[]
        n=len(nums)
        def dfs(depth,curr_list,begin,lastlayer_num,is_leaf):
            if is_leaf :
                if depth>2:
                    return_list.append(curr_list[:])
                return 

            #同层重复情况
            exist_layer_num=set()

            for i in range(begin,n):
                if nums[i]>=lastlayer_num and nums[i] not in exist_layer_num:
                    exist_layer_num.add(nums[i])
                    curr_list.append(nums[i])
                    dfs(depth+1,curr_list,i+1,nums[i],False)
                    curr_list.pop()
            dfs(depth+1,curr_list,0,0,True)
        dfs(0,[],0,-101,False)
        return return_list
# @lc code=end

