from typing import  List
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums.insert(0,1)
        nums.insert(len(nums),1)
        n=len(nums)
        dp=[ [0]*n for i in range(n)]
        for seperate_num in range(2,n):
            for i in range(0,n-seperate_num):
                j=i+seperate_num
                for k in range(i+1,j):
                    dp[i][j]=max(dp[i][j],dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j])
        return dp[0][n-1]

So=Solution()
nums=eval(input("please enter a list:"))
print(So.maxCoins(nums))

    
                
