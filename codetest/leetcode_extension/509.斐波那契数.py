#
# @lc app=leetcode.cn id=509 lang=python3
#
# [509] 斐波那契数
#

# @lc code=start
class Solution:
    def fib(self, n: int) -> int:
        f_0,f_1,f_2=0,1,0
        if n<1:
            return 0
        
        for _ in range(n-1):
            f_2=f_0+f_1
            f_0=f_1
            f_1=f_2
        return f_1
# @lc code=end

