#
# @lc app=leetcode.cn id=63 lang=python3
#
# [63] 不同路径 II
#
from typing import List

# @lc code=start
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m,n=len(obstacleGrid),len(obstacleGrid[0])
        #假设外面有一圈，其中出发点左上角有个可以动的位置其余全不能动
        dp=[0]*(n+1)
        dp[1]=1

        for row in obstacleGrid:
            for i in range(1,n+1):
                if row[i-1]==1:
                    dp[i]=0
                else:
                    dp[i]=dp[i-1]+dp[i]
        return dp[n]

# @lc code=end

