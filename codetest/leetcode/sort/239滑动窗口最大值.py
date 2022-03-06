import queue
from typing import List
# 使用递减栈 原因在于 当i<j ,若nums[j]>nums[i], 后续的滑动过程中有j在就不需要i，没有j时也没有i了。
class Solution:

    def queue_push(self,nums,lower_queue,new_index):
        while( lower_queue and nums[lower_queue[-1]]<=nums[new_index]):
            lower_queue.pop(-1)
        lower_queue.append(new_index)
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        lower_queue=[]
        return_list=[]
        for i in range(k-1):
            self.queue_push(nums,lower_queue,i)
        for left in range(len(nums)-k+1):

            self.queue_push(nums,lower_queue,left+k-1)
            while(lower_queue[0]<left):
                lower_queue.pop(0)
            return_list.append(nums[lower_queue[0]])
        return return_list

nums=eval(input("please enter score:"))
weight=eval(input("please enter weight:"))
cnt=0
for index,num in enumerate(nums):
    cnt+=num*weight[index]
print(cnt/sum(weight))