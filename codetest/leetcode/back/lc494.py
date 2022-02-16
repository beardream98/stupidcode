class Solution:
    def __init__(self) -> None:
        pass
    def findTargetSumWays(self, nums, target) -> int:
        cnt=0
        n=len(nums)

        def recu(i,temp_sum):
            nonlocal cnt
            if i==n:
                if temp_sum==target:
                    cnt+=1
                return 
            recu(i+1,temp_sum+nums[i])
            recu(i+1,temp_sum-nums[i])
        recu(0,0)
        return cnt 

s=Solution()
nums=eval(input("please enter a list:"))
target=eval(input("please enter a num:"))
print(s.findTargetSumWays(nums,target))


            

