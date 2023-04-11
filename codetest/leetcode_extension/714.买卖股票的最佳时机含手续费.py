#
# @lc app=leetcode.cn id=714 lang=python3
#
# [714] 买卖股票的最佳时机含手续费
#
from typing import List

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        profit=[0]*2
        profit[1]=float("-inf")
        profit_new=[0]*2
        for price in prices:
            profit_new[0]=max(profit[1]+price-fee,profit[0])
            profit_new[1]=max(profit[1],profit[0]-price)
            profit=profit_new
            profit_new=[0]*2
        return max(profit)
# @lc code=end

