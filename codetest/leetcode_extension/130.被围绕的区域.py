#
# @lc app=leetcode.cn id=130 lang=python3
#
# [130] 被围绕的区域
#
from typing import List

# @lc code=start
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m,n=len(board),len(board[0])
        visited=[ [0]*n for _ in range(m)]
        def dfs(i,j):
            if i<0 or i>=m or j<0 or j>=n:
                return 
            if visited[i][j]==0 and board[i][j]=="O":
                visited[i][j]=1
                dfs(i+1,j)
                dfs(i,j+1)       
                dfs(i-1,j)       
                dfs(i,j-1)       
        for i in range(m):
            dfs(i,0)
            dfs(i,n-1)
        for j in range(n):
            dfs(0,j)
            dfs(m-1,j)
        for i in range(m):
            for j in range(n):
                if board[i][j]=="O" and visited[i][j]==0:
                    board[i][j]="X"
        

            
# @lc code=end

