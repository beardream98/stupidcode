#
# @lc app=leetcode.cn id=46 lang=python3
#
# [46] 全排列
#
from typing import List

# @lc code=start
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        
        visited=[0]*n
        return_list=[]
        def dfs(index,curr_list):
            if index==n:
                return_list.append(curr_list[:])
            
            for i in range(n):
                if visited[i]==0:
                    curr_list.append(nums[i])
                    visited[i]=1
                    dfs(index+1,curr_list)
                    visited[i]=0
                    curr_list.pop()
        dfs(0,[])
        return return_list


# @lc code=end
if __name__=="__main__":
    nums = [1, 2, 3]
    solution = Solution()
    res = solution.permute(nums)
    print(res)
