#
# @lc app=leetcode.cn id=91 lang=python3
#
# [91] 解码方法
#

# @lc code=start


class Solution:
    def numDecodings(self, s: str) -> int:
        p_1,p_2,p_3=1,1,0
        n=len(s)
        if s[0]=="0":
            p_2=0


        for i in range(1,n):
            if s[i]!="0":
                p_3=p_2
            else:
                p_3=0
            char2=s[i-1]+s[i]
            if int(char2)>=10 and int(char2)<=26:
                p_3+=p_1
            p_1=p_2
            p_2=p_3
        return p_2 
# @lc code=end
if __name__=="__main__":
    so=Solution()
    s="27"
    res=so.numDecodings(s)
    print(res)
