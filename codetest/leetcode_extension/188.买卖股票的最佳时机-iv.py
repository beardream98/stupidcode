#
# @lc app=leetcode.cn id=188 lang=python3
#
# [188] 买卖股票的最佳时机 IV
#
from typing import List

# @lc code=start
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n=len(prices)
        profit=[ [0,float("-inf")] for _ in range(k+1)]
        new_profit=[ [0,float("-inf")] for _ in range(k+1)]

        for price in prices:
            for i in range(1,k+1):
                new_profit[i][0]=max(profit[i][0],profit[i][1]+price)
                new_profit[i][1]=max(profit[i][1],profit[i-1][0]-price)
            profit=new_profit
            new_profit=[ [0,float("-inf")] for _ in range(k+1)]
        return profit[k][0]
# @lc code=end
