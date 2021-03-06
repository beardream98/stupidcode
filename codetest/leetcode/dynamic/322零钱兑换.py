from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp=[amount+1]*(amount+1)
        dp[0]=[0]
        for j in range(1,amount+1):
            for coin in coins:
                if j-coin>=0:
                    dp[j]=min(dp[j-coin]+1,dp[j])
        return dp[amount] if dp[amount]<amount+1 else -1

So=Solution()
coins=eval(input("please enter a list:"))
amount=eval(input("please enter a number:"))

print(So.coinChange(coins,amount))
            