from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<=1:return 0
        dp=[ [0]*3 for i in range(len(prices))]
        dp[0][0],dp[0][1],dp[0][2]=0,-prices[0],0
        for i in range(1,len(prices)):
            dp[i][0]=max(dp[i-1][0],dp[i-1][2])
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i])
            dp[i][2]=dp[i-1][1]+prices[i]
        return max(dp[-1][0],dp[-1][2])
So=Solution()
prices=eval(input("please enter a list:"))
print(So.maxProfit(prices))
        
