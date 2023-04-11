#
# @lc app=leetcode.cn id=494 lang=python3
#
# [494] 目标和
#
from typing import List

# @lc code=start
from collections import defaultdict

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # target=(sum-neg)-neg
        # neg=(sum-target)/2
        neg=sum(nums)-target
        if neg<0 or neg%2!=0:
            return 0
        neg=neg//2
        dp=[0]*(neg+1)
        dp[0]=1
        for i in range(len(nums)):
            for j in reversed(range(nums[i],neg+1)):
                dp[j]=dp[j-nums[i]]+dp[j]
        return dp[neg]


# @lc code=end

##方法一 通过保存每一次可能结果进行
# def findTargetSumWays(self, nums: List[int], target: int) -> int:

#     num_info=defaultdict(int)
#     temp_info=defaultdict(int)
#     num_info[0]=1
#     for num in nums:
#         for key,value in num_info.items():
#             a,b=key+num,key-num
#             temp_info[a]+=value
#             temp_info[b]+=value
#         num_info=temp_info
#         temp_info=defaultdict(int)
#     return num_info[target]
if __name__=="__main__":
    nums=[1,1,1,1,1]
    target=3
    so=Solution()
    res=so.findTargetSumWays(nums,target)
    print(res)