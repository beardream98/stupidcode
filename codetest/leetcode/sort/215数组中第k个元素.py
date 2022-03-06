from turtle import right
from typing import List
class Solution1:
    # 快排
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def parti(nums,left_edge):
            i,j=0,len(nums)-1
            while(i<j):
                #条件判断
                for t in reversed(range(i,j+1)):
                    if nums[t]<nums[i]:
                        temp=nums[t]
                        nums[t]=nums[i]
                        nums[i]=temp
                        break
                j=t
                for q in range(i,j+1):
                    if nums[q]>nums[j]:
                        temp=nums[q]
                        nums[q]=nums[j]
                        nums[j]=temp
                        break
                i=q
            return nums,i+left_edge
        #一步 相对坐标到绝对坐标转换
        left_edge=0
        right_edge=len(nums)-1
        nums,index=parti(nums,left_edge)


        while(index+1!=k):
            if index+1>k:
                right_edge=index-1
                nums[left_edge:right_edge+1],index=parti(nums[left_edge:right_edge+1],left_edge)
            else:
                left_edge=index+1
                nums[left_edge:right_edge+1],index=parti(nums[left_edge:right_edge+1],left_edge)

        return nums[index]



def down_node(nums,index,n):
    left_index=2*index+1
    right_index=2*index+2
    if right_index<n:
        change_index=left_index if nums[left_index]>nums[right_index] else right_index
    elif left_index<n:
        change_index=left_index
    else:
        return 
    if nums[change_index]>nums[index]:
        temp=nums[change_index]
        nums[change_index]=nums[index]
        nums[index]=temp
        down_node(nums,change_index,n)
def create_heap(nums):
    
    big_heap=nums.copy()
    n=len(big_heap)
    if n==0: return []
    k=1
    while(2**k-1<=n):
        k+=1
    k=2**(k-1)-1
    for i in reversed(range(0,k)):
        down_node(big_heap,i,n)
    return big_heap
def output_heap(bigheap):
    n=len(bigheap)
    if n==0: return None
    output=bigheap[0]
    bigheap[0]=bigheap[n-1]
    #使用切片赋值方式在函数内无法正确修改
    bigheap.pop()
    down_node(bigheap,0,n-1)
    return output
nums=[3,2,1,5,6,4] 
k=2

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        bigheap=create_heap(nums)
        output=-1
        for i in range(k):
            output=output_heap(bigheap)
        return output
So=Solution()
print(So.findKthLargest(nums,k))



