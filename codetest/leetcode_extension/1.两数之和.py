#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
#array | hash-table

from typing import List
from collections import defaultdict

class Solution1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=defaultdict(list)
        for index,num in enumerate(nums):
            hashmap[num].append(index)
        for index,num in enumerate(nums):
            #两数相同时必须是不同index
            # 解唯一
            minus_num=target-num
            if (minus_num)!=num and hashmap[minus_num]:
                return [index,hashmap[minus_num][0]]
            if (minus_num)==num and len(hashmap[minus_num])>1:
                return [index]+[ i for i in hashmap[minus_num] if i!=index]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 若target 由i 和j 组成，若i>j 当遍历到i时 查询失效但是到j时查询成功
        # 只需一遍且避免了 x找到同一index 的x情况
        hashmap={}
        for index,num in enumerate(nums):
            minus_num=target-num
            if hashmap.get(minus_num,None)==None:
                hashmap[num]=index
            else:
                return [hashmap[minus_num],index]

# @lc code=end
from typer import open_file
with open_file("testwrite","r") as file_object:
    readlines=file_object.readline()
    nums=eval(readlines.strip())
    target=eval(file_object.readline().strip())
So=Solution()
print(So.twoSum(nums,target))