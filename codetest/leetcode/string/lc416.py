from typing import List
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target=sum(nums)
        n=len(nums)
        if target%2!=0:
            return False
        target=target/2
        flag=False
        def bfs(i,target):
            nonlocal flag
            if flag==True:
                return 
            if nums[i]==target:
                flag=True
            elif nums[i]<target:
                for k in range(i+1,n):
                    bfs(k,target-nums[i])
        bfs(0,target)
        return flag

So=Solution()
nums=eval(input("please enter a list:"))
print(So.canPartition(nums))