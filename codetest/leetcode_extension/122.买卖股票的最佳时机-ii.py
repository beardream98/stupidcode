#
# @lc app=leetcode.cn id=122 lang=python3
#
# [122] 买卖股票的最佳时机 II
#
from typing import List

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        dp=[[0]*2 for _ in range(n+1)]
        dp[0][0]=float("-inf")
        for i in range(1,n+1):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i-1])
            dp[i][1]=max(dp[i-1][0]+prices[i-1],dp[i-1][1])
        
        return dp[n][1]

# @lc code=end

