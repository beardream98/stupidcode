#
# @lc app=leetcode.cn id=198 lang=python3
#
# [198] 打家劫舍
#
from typing import List

# @lc code=start
class Solution:
    def rob(self, nums: List[int]) -> int:
        n=len(nums)
        dp=[[0]*2 for _ in range(n+1)]
        for i in range(1,n+1):
            dp[i][0]=dp[i-1][1]+nums[i-1]
            dp[i][1]=max(dp[i-1][0],dp[i-1][1])
        return max(dp[n][0],dp[n][1])
        

# @lc code=end

