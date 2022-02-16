import math
def numSquares(n) :
    f=[0]*(n+1)
    for i in range(1,n+1):
        min_num=float("inf")
        for j in range(1,int(math.sqrt(i))+1):
            index=i-j**2
            if f[index]<min_num:
                min_num=f[index]
        f[i]=1+min_num
    return f[n]
print(numSquares(12))

