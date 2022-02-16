def solution(N):
    # write your code in Python 3.6
    num_stack=[]
    num,n=N+1,0
    while(num!=0):
        num_stack.append(num%10)
        num=num//10
        n+=1
    i=0
    while(i<n-1):
        if num_stack[i]==num_stack[i+1]:
            num_stack[i]+=1
            if num_stack[i]==10:
                k=i
                while(k<n and num_stack[k]==10):
                    num_stack[k]=0
                    if k==n-1:
                        continue
                    num_stack[k+1]+=1
                    k+=1
            for j in range(i):
                num_stack[j]=0
            i=0
        else:
            i+=1
    cnt=0
    if num_stack[-1]==0:
        num_stack[-1]=10
    for num in reversed(num_stack):
        
        cnt+=num
        cnt*=10
    return int(cnt/10)
N=eval(input("enter a number:"))
print(solution(N))