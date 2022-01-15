def canJump( nums) :
    # 为0的下标
    zero_list=[]
    n=len(nums)
    for i in range(n-1):
        if nums[i]==0:
            zero_list.append(i)
    if len(zero_list)==0:
        return True
    i,step=-1,1
    for zero_index in zero_list:
        while(step<=zero_index-i and i<zero_index):
            i+=1
            step=nums[i]
        if i==zero_index:
            return False
    return True
#贪心法
def canJump1( nums) :
    #不断更新最远距离
    i,n=0,len(nums)
    max_index=nums[i]
    while(i<=max_index and i<n-1):
        if i+nums[i]>max_index:
            max_index=i+nums[i]
        i+=1
    if max_index>=n-1:
        return True
    else:
        return False







nums=eval(input("please enter a list:"))
print(canJump1(nums))

            
