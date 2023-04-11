#
# @lc app=leetcode.cn id=279 lang=python3
#
# [279] 完全平方数
#

# @lc code=start
from math import sqrt,floor

class Solution:
    def numSquares(self, n: int) -> int:
        dp=[ i for i in range(n+1)]
        for i in range(1,floor(sqrt(n)+1)):
            dp[i*i]=1
        for i in range(2,n+1):
            for j in range(1,floor(sqrt(i)+1)):
                dp[i]=min(dp[i],1+dp[i-j**2])
            
        return dp[n]

# @lc code=end

if __name__=="__main__":
    so=Solution()
    n=7691
    res=so.numSquares(n)
    print(res)