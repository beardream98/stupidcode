#
# @lc app=leetcode.cn id=22 lang=python3
#
# [22] 括号生成
#
from typing import List

# @lc code=start
class Solution:
    def concat_list(self,list1,list2):
        m,n=len(list1),len(list2)
        return_list=[]
        for i in range(m):
            for j in range(n):
                return_list.append(list1[i]+list2[j])
        return return_list
        
    def generateParenthesis(self, n: int) -> List[str]:
        all_l=[["()"]]
        new_l=[]
        for i in range(1,n):
            for j in range(0,i):
                k=i-j
                new_l+=self.concat_list(all_l[j],all_l[k-1])
            for t in all_l[i-1]:
                new_l.append("("+t+")")
            all_l.append(list(set(new_l[:])))
            new_l=[]
        return all_l[-1]
        

                

# @lc code=end
