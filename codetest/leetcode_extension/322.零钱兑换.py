#
# @lc app=leetcode.cn id=322 lang=python3
#
# [322] 零钱兑换
#
from typing import List

# @lc code=start
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n=amount
        dp=[-1]*(n+1)
        dp[0]=0
        for i in range(1,n+1):
            for coin in coins:
                if i-coin>=0 and dp[i-coin]!=-1:
                    if dp[i]==-1:
                        dp[i]=dp[i-coin]+1
                    else:
                        dp[i]=min(dp[i],dp[i-coin]+1)
        return dp[n]

        

# @lc code=end

if __name__=="__main__":
    coins=[2]
    amount=3
    so=Solution()
    res=so.coinChange(coins,amount)
    print(res)