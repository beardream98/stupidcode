from tkinter.tix import TList
from typing import List
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m,n=len(board),len(board[0])
        t=len(word)
        visited=[ [0]*n for i in range(m)]
        flag=False
        def dfs(index,i,j):
            nonlocal flag
            if index==t-1 or flag==True:
                flag=True
                return
            #可能存在越界
            visited[i][j]=1
            if  j-1>=0 and  visited[i][j-1]==0 and word[index+1]==board[i][j-1]:
                dfs(index+1,i,j-1)
            if i-1>=0  and  visited[i-1][j]==0 and word[index+1]==board[i-1][j]:
                dfs(index+1,i-1,j)
            if i+1<m  and  visited[i+1][j]==0 and word[index+1]==board[i+1][j]:
                dfs(index+1,i+1,j)
            if j+1<n and  visited[i][j+1]==0 and word[index+1]==board[i][j+1]:
                dfs(index+1,i,j+1)
            visited[i][j]=0
        for i in range(m):
            for j in range(n):
                if flag==False and word[0]==board[i][j]:
                    dfs(0,i,j)
        return flag
So=Solution()
board=eval(input("please enter a matrix :"))
word=input("plase enter a string :")

print(So.exist(board,word))

        
