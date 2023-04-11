#
# @lc app=leetcode.cn id=42 lang=python3
#
# [42] 接雨水
#

# @lc code=start
from typing import List
class Solution:
    def trap(self, height: List[int]) -> int:
        """
        每个位置能装多少水，取决于左边最大的木桶边和右边最大的木桶边较小的那个
        """
        n=len(height)
        heightMax_left,heightMax_right=[0]*n,[0]*n
        heightMax_left[0],heightMax_right[n-1]=height[0],height[n-1]


        for i in range(1,n):
            j=n-1-i
            heightMax_left[i]=max(height[i],heightMax_left[i-1])
            heightMax_right[j]=max(height[j],heightMax_right[j+1])
        cnt=0
        for i in range(n):
            cnt+=min(heightMax_left[i],heightMax_right[i])-height[i]
        return cnt
# @lc code=end

So=Solution()
height=[4,2,0,3,2,5]
print(So.trap(height))

