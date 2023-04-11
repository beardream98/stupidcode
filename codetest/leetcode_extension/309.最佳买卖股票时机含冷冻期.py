#
# @lc app=leetcode.cn id=309 lang=python3
#
# [309] 最佳买卖股票时机含冷冻期
#
from typing import List
# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit=[0]*3
        profit[1]=float("-inf")
        profit_new=[0]*3
        for price in prices:
            profit_new[0]=max(profit[2],profit[0])
            profit_new[1]=max(profit[1],profit[0]-price)
            profit_new[2]=profit_new[1]+price
            profit=profit_new
            profit_new=[0]*3
        return max(profit)
        

# @lc code=end

