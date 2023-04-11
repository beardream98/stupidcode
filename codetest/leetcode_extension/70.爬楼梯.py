#
# @lc app=leetcode.cn id=70 lang=python3
#
# [70] 爬楼梯
#

# @lc code=start
class Solution:
    def climbStairs(self, n: int) -> int:
        p_1,q_1,p_2,q_2,p_3,q_3=1,0,1,1,0,0
        if n<=1:
            return 1
        
        for _ in range(n-2):
            p_3=p_2+q_2
            q_3=p_1+q_1
            p_1,q_1=p_2,q_2
            p_2,q_2=p_3,q_3
        return p_2+q_2
# @lc code=end

if __name__ =="__main__":
    so=Solution()
    n=4
    res=so.climbStairs(n)
    print(res)
