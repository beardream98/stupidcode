def subsets(nums) :
    n=len(nums)
    subset=[]
    for i in range(2**n):
        temp_set=[]
        k,w=i,0
        while(k!=0):
            t,k=k%2,k//2
            if t==1:
                temp_set.append(nums[w])
            w+=1
        subset.append(temp_set)
    return subset

nums=eval(input("please enter a subset:"))
print(subsets(nums))