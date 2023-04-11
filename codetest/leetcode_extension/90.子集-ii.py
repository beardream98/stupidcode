#
# @lc app=leetcode.cn id=90 lang=python3
#
# [90] 子集 II
#
from typing import List
# @lc code=start
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        return_list=[]
        n=len(nums)
        #sort 控制向下不重复
        nums.sort()
        
        def dfs(path,add_empty,begin):
            if add_empty:
                return_list.append(path[:])
                return 
            #控制层不重复
            layer_exist={}
            #begin 控制向下不重复
            for i in range(begin,n):
                if layer_exist.get(nums[i],-1)==-1:
                    layer_exist[nums[i]]=1
                    path.append(nums[i])
                    dfs(path,False,i+1)
                    path.pop()
            dfs(path,True,begin)
        dfs([],False,0)
        return return_list
# @lc code=end
if __name__=="__main__":
    so=Solution()
    nums=[4,1,4,4,4]
    res=so.subsetsWithDup(nums)
    print(res)

