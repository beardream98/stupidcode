from typing import List

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m,n=len(matrix),len(matrix[0])
        dp=[ [0]*n for i in range(m)]
        for i in range(m):
            for j in range(n):
                if i==0 or j==0:
                    dp[i][j]=int(matrix[i][j])
                    continue
                if int(matrix[i][j])!=0:
                    dp[i][j]=min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1
        return max([max(row) for row in dp])**2
So=Solution()
matrix=eval(input("please enter a matrix:"))
print(So.maximalSquare(matrix))