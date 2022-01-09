def longestValidParentheses1(s):

    stack_flag=[]
    stack_index=[]
    s=s+"*"
    for index,char in enumerate(s):
        if char==")" and len(stack_flag)!=0 and stack_flag[-1]=="(":
            stack_flag.pop()
            stack_index.pop()
        else:
            stack_flag.append(char)
            stack_index.append(index)
    max_length=stack_index[0]
    if len(stack_index)==1:
        return max_length
    for i in range(1,len(stack_index)):
        ill_num=stack_index[i]
        last_num=stack_index[i-1]
        if ill_num-last_num-1>max_length:
            max_length=ill_num-last_num-1
    return max_length
def longestValidParentheses2(s):
    #动态规划法
    s="**"+s
    dp=len(s)*[0]
    for i in range(2,len(s)):
        if s[i]==")" and s[i-1]=="(":
            dp[i]=dp[i-2]+2
        elif s[i]==")" and s[i-1]==")" and s[i-dp[i-1]-1]=="(":
            dp[i]=dp[i-1]+dp[i-dp[i-1]-2]+2
    max_length=max(dp)
    return max_length
    
    

s=input("enter chars:")
print(longestValidParentheses2(s))