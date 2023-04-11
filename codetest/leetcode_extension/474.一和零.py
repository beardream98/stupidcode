#
# @lc app=leetcode.cn id=474 lang=python3
#
# [474] 一和零
#
from typing import List
# @lc code=start
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        #将物品价值设为1，转换为0/1背包问题。（多了一维）
        dp=[(n+1)*[0] for _ in range(m+1)]
        #需要空的状态

        for str in strs:
            zero_num=str.count("0")
            one_num=str.count("1")
            for i in reversed(range(zero_num,m+1)):
                for j in reversed(range(one_num,n+1)):
                    dp[i][j]=max(dp[i][j],dp[i-zero_num][j-one_num]+1)
        return dp[m][n]


# @lc code=end
if __name__=="__main__":
    so=Solution()
    strs =  ["10", "0001", "111001", "1", "0"]
    m = 5
    n = 3
    res=so.findMaxForm(strs,m,n)
    print(res)
