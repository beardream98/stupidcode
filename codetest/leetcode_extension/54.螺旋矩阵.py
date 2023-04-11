#
# @lc app=leetcode.cn id=54 lang=python3
#
# [54] 螺旋矩阵
#
from typing import List

# @lc code=start
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m,n=len(matrix),len(matrix[0])
        i_s,j_s,i_e,j_e=0,0,m-1,n-1
        return_list=[]
        while(i_e>i_s and j_e>j_s):
            return_list+=[ matrix[i_s][t] for t in range(j_s,j_e+1)]
            return_list+=[ matrix[t][j_e] for t in range(i_s+1,i_e+1)]
            
            return_list+=reversed([ matrix[i_e][t] for t in range(j_s,j_e)])
            return_list+=reversed([ matrix[t][j_s] for t in range(i_s+1,i_e)])

            i_e-=1
            i_s+=1
            j_e-=1
            j_s+=1
        if i_e==i_s and j_e>=j_s:
            return_list+=[ matrix[i_s][t] for t in range(j_s,j_e+1)]
        elif j_e==j_s and i_e>=i_s:
            return_list+=[ matrix[t][j_e] for t in range(i_s,i_e+1)]
        return return_list

# @lc code=end
matrix=[[1,2,3],[4,5,6]]




