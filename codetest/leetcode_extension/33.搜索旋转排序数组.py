#
# @lc app=leetcode.cn id=33 lang=python3
#
# [33] 搜索旋转排序数组
#
from typing import List

# @lc code=start
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n=len(nums)
        left,right=0,n-1
        while( left<=right):
            mid=(left+right)//2
            if (nums[mid]==target):
                return mid 
            if nums[mid]>=nums[left]:
                #左边有序
                if target>=nums[left] and target<=nums[mid]:
                    right=mid-1
                else:
                    left=mid+1
            elif  nums[mid]<nums[left]:
                #右边有序
                if target>=nums[mid] and target<=nums[right]:
                    left=mid+1
                else:
                    right=mid-1
        
        return -1
# @lc code=end

nums=[3,1]
target=1
So=Solution()
print(So.search(nums,target))