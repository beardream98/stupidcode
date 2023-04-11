#
# @lc app=leetcode.cn id=416 lang=python3
#
# [416] 分割等和子集
#
from typing import List

# @lc code=start
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target=sum(nums)
        n=len(nums)
        if target%2!=0 or n==1:
            return False
        dp=(target+1)*[0]
        dp[0]=1

        for num in nums:
            for j in reversed(range(num,target+1)):
                dp[j]=max(dp[j-num],dp[j])


        return dp[target//2]==1
        

# @lc code=end
if __name__=="__main__":
    nums=[1,2,3,5]
    So=Solution()
    res=So.canPartition(nums)
    print(res)
