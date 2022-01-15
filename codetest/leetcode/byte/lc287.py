def findDuplicate( nums) :
    n=len(nums)-1
    l,r,ans=1,n,-1
    while(l<r):
        mid=(l+r)//2
        cnt=0
        for i in range(n+1):
            if nums[i]<=mid:
                cnt+=1
        if cnt<=mid:
            l=mid+1
        else:
            r=mid-1
            ans=mid
        
    return ans
        

nums=eval(input("please enter a list:"))
print(findDuplicate(nums))

