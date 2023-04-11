#
# @lc app=leetcode.cn id=354 lang=python3
#
# [354] 俄罗斯套娃信封问题
#
from typing import List
# @lc code=start
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes=sorted(envelopes,key=lambda x:(x[0],x[1]))

        n=len(envelopes)
        dp=[1]*(n)
        for i,envelope in enumerate(envelopes):
            for j in range(i):
                if envelope[0]>envelopes[j][0] and envelope[1]>envelopes[j][1]:
                    dp[i]=max(dp[i],dp[j]+1)
        return max(dp)

# @lc code=end

if __name__=="__main__":
    so=Solution()
    envelopes=[[2,100],[3,200],[4,300],[5,500],[5,400],[5,250],[6,370],[6,360],[7,380]]
    res=so.maxEnvelopes(envelopes)
    print(res)