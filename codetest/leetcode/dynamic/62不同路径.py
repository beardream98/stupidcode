class Solution1:
    def uniquePaths(self, m: int, n: int) -> int:
        cnt=0
        #深搜 超时
        def dfs(i,j):
            nonlocal cnt
            if i==m-1 and j==n-1:
                cnt+=1
                return 
            if i<m:
                dfs(i+1,j)
            if j<n:
                dfs(i,j+1)
        dfs(0,0)
        return cnt

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        path_count=[[0]*n for i in range(m)]
        for i in range(m):
            for j in range(n):
                if i==0 or j==0:
                   path_count[i][j]=1
                else:
                    path_count[i][j]=path_count[i-1][j]+path_count[i][j-1]
        return path_count[m-1][n-1]

So=Solution()
m,n=eval(input('please enter m:')),eval(input("please enter n:"))
print(So.uniquePaths(m,n))
