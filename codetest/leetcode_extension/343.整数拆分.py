#
# @lc app=leetcode.cn id=343 lang=python3
#
# [343] 整数拆分
#

# @lc code=start
class Solution:
    def integerBreak(self, n: int) -> int:
        # 多个和 化解为两个和问题
        #dp 假设是包括不拆情况下乘积最大，需要一个记录值保存不拆的最大值
        dp=[1]*(n+1)
        for i in range(2,n+1):
            for j in range(1,i//2+1):
                dp[i]=max(dp[i],dp[j]*dp[i-j])
            temp=dp[i]

            dp[i]=max(dp[i],i)

        return temp

# @lc code=end

if __name__=="__main__":
    so=Solution()
    n=10
    res=so.integerBreak(n)
    print(res)
