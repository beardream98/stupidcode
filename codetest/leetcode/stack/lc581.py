def findUnsortedSubarray(nums) :
    n=len(nums)
    left,right=0,-1
    maxn,minn=float("-inf"),float("inf")
    for i in range(n):
        if nums[i]>maxn:
            maxn=nums[i]
        else:
            right=i
    for j in range(n-1,-1,-1):
        if nums[j]<minn:
            minn=nums[j]
        else:
            left=j
    return right-left+1


    


nums=[5,4,3,2,1]
print(findUnsortedSubarray(nums))
            