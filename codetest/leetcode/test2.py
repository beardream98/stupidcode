def canJump(nums) :
    n=len(nums)
    i,j=0,0
    new_i=0
    max_index=i+nums[i]
    new_max=max_index
    while(max_index<n-1):
        while (j<max_index):
            if nums[j]+j>max_index:
                new_max=nums[j]+j
                new_i=j
        if new_max==max_index:
            return False
        else:
            i=new_i
            max_index=new_max
    return True

nums=[2,3,1,1,4]
canJump(nums)