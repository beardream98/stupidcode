#
# @lc app=leetcode.cn id=60 lang=python3
#
# [60] 排列序列
#

# @lc code=start
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        return_list=[]

        visited=n*[0]
        def dfs(depth,curr_list):
            if depth==n:
                return_list.append(curr_list[:])
                return 
            for i in range(n):
                if visited[i]!=1:
                    visited[i]=1
                    curr_list+=f"{i+1}"
                    dfs(depth+1,curr_list)
                    visited[i]=0
                    curr_list=curr_list[:-1]


        dfs(0,"")
        return return_list[k-1]
# @lc code=end

