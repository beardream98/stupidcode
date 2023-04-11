#
# @lc app=leetcode.cn id=15 lang=python3
#
# [15] 三数之和
#
from typing import List

# @lc code=start
# 双指针指向i，j 列表有序时能较好排查结果
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        return_list=[]
        nums=sorted(nums)
        n=len(nums)
        if n<3:
            return []
        for i in range(n-2):
            if i!=0 and (nums[i]==nums[i-1] or nums[i]>0):
                continue
            p,q=i+1,n-1
            while(p<q):
                if nums[p]+nums[q]==-1*nums[i]:
                    return_list.append([nums[i],nums[p],nums[q]])
                    p,q=p+1,q-1
                    while(p<q and nums[p]==nums[p-1] ):
                        p=p+1
                    while(p<q and nums[q]==nums[q+1] ):
                        q=q-1
                elif nums[p]+nums[q]>-1*nums[i]:
                    q-=1
                else:
                    p+=1
        return return_list
# @lc code=end
nums=[0,0,0]
So=Solution()
print(So.threeSum(nums))
