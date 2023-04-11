#
# @lc app=leetcode.cn id=5 lang=python3
#
# [5] 最长回文子串
#

# @lc code=start
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n=len(s)
        ispalindrome=[ [0]*n for i in range(n)]
        ispalindrome[0][0]=1
        for j in range(1,n):
            for i in range(j+1):
                if s[i]==s[j] and ( i+1>j-1 or ispalindrome[i+1][j-1]==1):
                    ispalindrome[i][j]=1
                else:
                    ispalindrome[i][j]=0
        max_len=0
        return_string=""
        for j in range(n):
            for i in range(j+1):
                if ispalindrome[i][j] and j-i+1>max_len:
                    max_len=max(max_len,j-i+1)
                    return_string="".join(s[i:j+1])
        return return_string
# @lc code=end


