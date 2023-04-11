#
# @lc app=leetcode.cn id=120 lang=python3
#
# [120] 三角形最小路径和
#
from typing import List

# @lc code=start
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        #无后效性 i+1的最优情况由i的最优情况决定
        n=len(triangle[-1])
        dp=[0]*n
        dp[0]=triangle[0][0]
        for row in triangle[1:]:
            row_len=len(row)
            dp[row_len-1]=dp[row_len-2]+row[row_len-1]

            for j in reversed(range(1,row_len-1)):
                dp[j]=min(dp[j],dp[j-1])+row[j]
            
            dp[0]+=row[0]
        return min(dp)

# @lc code=end
if __name__=="__main__":
    triangle=[[2],[3,4],[6,5,7],[4,1,8,3]]
    so=Solution()
    res=so.minimumTotal(triangle)
    print(res)
