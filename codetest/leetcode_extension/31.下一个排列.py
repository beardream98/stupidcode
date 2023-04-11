#
# @lc app=leetcode.cn id=31 lang=python3
#
# [31] 下一个排列
#
from typing import List

# @lc code=start
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        j=n-2
        while(j>=0):
            if nums[j]<nums[j+1]:
                #找到相应小数
                break
            else:
                j-=1
        if j==-1:
            nums[:]=sorted(nums)
        else:
            for i in reversed(range(j+1,n)):
                if nums[i]>nums[j]:
                    break
            temp=nums[j]
            nums[j],nums[i]=nums[i],temp
            nums[j+1:]=sorted(nums[j+1:])
# @lc code=end
So=Solution()
nums=[3,2,1]
So.nextPermutation(nums)