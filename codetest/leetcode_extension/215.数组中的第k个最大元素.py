#
# @lc app=leetcode.cn id=215 lang=python3
#
# [215] 数组中的第K个最大元素
#
from typing import List

# @lc code=start
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def parti(nums,i,j):
            #边界条件
            while(i<j):
                while(i<j):
                    if nums[j]<nums[i]:
                        temp=nums[j]
                        nums[j]=nums[i]
                        nums[i]=temp
                        break
                    j-=1
                while(i<j):
                    if nums[i]>nums[j]:
                        temp=nums[j]
                        nums[j]=nums[i]
                        nums[i]=temp
                        break
                    i+=1
            return i
        i,j,index=0,len(nums)-1,-1
        new_k=len(nums)-k
        while new_k!=index:
            index=parti(nums,i,j)
            if index<new_k:
                i=index+1
            elif index>new_k:
                j=index-1

        return nums[new_k]
# @lc code=end

 


