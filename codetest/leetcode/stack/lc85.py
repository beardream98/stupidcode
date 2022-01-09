def maximalRectangle(matrix):
    if len(matrix)==0:
        return 0
    m,n=len(matrix),len(matrix[0])
    height=[[0]*n for i in range(m+1)]
    matrix.append(["0"]*n)
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j]!="0" and j!=0:
                height[i][j]=height[i][j-1]+int(matrix[i][j])
            else:
                height[i][j]=int(matrix[i][j])

    max_cnt=0
    for j in range(n):
        stack,i=[],0
        while (i<m+1):
            if len(stack)==0 or height[stack[-1]][j]<=height[i][j]:
                stack.append(i)
                i+=1
            else:
                mid_index=stack.pop()
                if len(stack)==0:
                    left_index=-1
                else:
                    left_index=stack[-1]
                cnt=(i-left_index-1)*height[mid_index][j]
                if cnt>max_cnt:
                    max_cnt=cnt
    return max_cnt
matrix=eval(input("please enter matrix :"))

print(maximalRectangle(matrix))