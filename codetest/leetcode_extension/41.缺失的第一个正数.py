#
# @lc app=leetcode.cn id=41 lang=python3
#
# [41] 缺失的第一个正数
#
from typing import  List
class Solution1:
    def firstMissingPositive(self, nums: List[int]) -> int:
        #原地哈希
        n=len(nums)
        for i in range(n):
            if nums[i]<=0:
                nums[i]=n+1
        for j in range(n):
            index=abs(nums[j])
            if abs(index)<=n and nums[index-1]>0:
                nums[index-1]*=-1
        for t in range(n):
            if nums[t]>0:
                return t+1
        return n+1
# @lc code=start
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # 置换正确位置
        n,i=len(nums),0
        while(i<n):
            if nums[i]<=0 or nums[i]>n or nums[i]==nums[nums[i]-1]:
                i+=1
            else:
                k=nums[i]-1
                temp=nums[i]
                nums[i]=nums[k]
                nums[k]=temp
        for i in range(n):
            if nums[i]!=i+1:
                return i+1
        return n+1

# @lc code=end

