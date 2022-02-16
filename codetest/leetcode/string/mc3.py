def solution(A):
    # write your code in Python 3.6
    more_node,less_node=[],[]
    for i in range(3):
        for j in range(3):
            if A[i][j]>1:
                more_node.append([i,j])
            elif A[i][j]==0:
                less_node.append([i,j])
    min_cnt=9999
    def BFS(A,i,j,cnt):
        nonlocal min_cnt
        if sum([A[node[0]][node[1]] for node in less_node])==len(less_node):
            if cnt<min_cnt:
                min_cnt=cnt
        if A[i][j]>1:
            for node in less_node:
                if A[node[0]][node[1]]==0:
                    A[i][j]-=1
                    A[node[0]][node[1]]=1
                    cnt+=abs(i-node[0])+abs(j-node[1])
                    BFS(A,i,j,cnt)
                    cnt-=abs(i-node[0])+abs(j-node[1])
                    A[i][j]+=1
                    A[node[0]][node[1]]=0
        if A[i][j]==1:
            for node in more_node:
                if A[node[0]][node[1]]>1:
                    BFS(A,node[0],node[1],cnt)
    for node in more_node:
        BFS(A,node[0],node[1],0)
        break
    return min_cnt
A=eval(input("enter a matrix:"))
print(solution(A))

# [[1,0,1],[1,3,0],[2,0,1]]
