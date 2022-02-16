def numIslands(grid):
    m,n=len(grid),len(grid[0])
    view_position=[ [0]*n for i in range(m)]
    cnt=0
    def DFS(i,j):
        if grid[i][j]==str(0):
            view_position[i][j]=1
            return 0
        view_position[i][j]=1
        if j+1<n and view_position[i][j+1]==0:
            DFS(i,j+1)
        if i+1<m and view_position[i+1][j]==0:
            DFS(i+1,j)
        return 1
    for i in range(m):
        for j in range(n):
            if view_position[i][j]!=1:
               cnt+=DFS(i,j)
    return cnt
grid=[["1","1","1"],["0","1","0"],["1","1","1"]]

print(numIslands(grid))
    