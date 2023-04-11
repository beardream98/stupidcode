
m=int(input())
n=int(input())
def max_value(n,m,v,w):
    dp=[0]*(m+1)
    for i in range(i,n+1):
        for j in range(m,v[i]-1,-1):
            dp[j]=max(dp[j],dp[j-v[i]]+w[i])
    
    return dp[-1]
for i in range()