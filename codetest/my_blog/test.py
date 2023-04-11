# n,m,x,y = map(int,input().split(" "))
# matrix1 = []
# for _ in range(n):
#     matrix1.append(list(map(int,input().split(" "))))
n,m,x,y = 3,3,2,2
matrix1 = [[1,2,0],[-2,2,1],[2,1,2]]
row_sum = [[0]*m for _ in range(n)]
for i in range(n):
    cnt = 1
    for j in range(m):
        if matrix1[i][j] == 0:
            cnt *= 0.001
        else:
            cnt *= matrix1[i][j]
        row_sum[i][j] = cnt

matrix_sum = [[0]*m for _ in range(n)]
for j in range(m):
    matrix_sum[0][j] = row_sum[0][j]
for i in range(1,n):
    for j in range(m):
        matrix_sum[i][j] = matrix_sum[i-1][j]*row_sum[i][j]
max_sum = 0
max_x = 0
max_y = 0
for i in range(n-x+1):
    for j in range(n-y+1):
        if j == 0:
            cnt1 = 1
        else:
            cnt1 = matrix_sum[i+x-1][j-1]
        if i == 0:
            cnt2 = 1
        else:
            cnt2 = matrix_sum[i-1][j+x-1]
        if i!=0 and j!=0:
            cnt3 = matrix_sum[i-1][j-1]
        else:
            cnt3 = 1
        temp_sum = matrix_sum[i+x-1][j+x-1]* cnt3 / cnt1 / cnt2
        if i==0 and j==0:
            max_sum=temp_sum
            continue
        if temp_sum > max_sum:
            max_x,max_y = i,j
            max_sum = temp_sum

print(max_x+1,max_y+1)


