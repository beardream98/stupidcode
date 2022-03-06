from typing import List
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp=len(nums)*[1]
        for i in range(1,len(nums)):
            max_index,max_num=-1,-1
            for j in range(0,i):
                if nums[j]<nums[i] and dp[j]>max_num :
                    max_num,max_index=dp[j],j
            if max_index!=-1:
                dp[i]=dp[max_index]+1
        return max(dp)


So=Solution()
nums=eval(input("please enter a list:"))
print(So.lengthOfLIS(nums))
            