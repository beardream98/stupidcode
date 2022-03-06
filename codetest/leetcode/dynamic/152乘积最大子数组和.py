from typing import List
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res,min_n,max_n=nums[0],nums[0],nums[0]
        for num in nums[1:]:
            if num>0:
                min_n=min(num,min_n*num)
                max_n=max(num,max_n*num)
            else:
                pre_min,pre_max=max_n*num,min_n*num
                min_n=min(num,pre_min)
                max_n=max(num,pre_max)
            res=max(max_n,res)
        return res


So=Solution()
nums=eval(input("plase enter a list:"))
print(So.maxProduct(nums))
