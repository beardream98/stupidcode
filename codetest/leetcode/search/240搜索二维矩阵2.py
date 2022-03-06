from typing import List
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        def binarysearch(nums,target):
            n=len(nums)
            i,j=0,n-1
            while(i<=j):
                mid=(i+j)//2
                if nums[mid]==target:
                    return True
                elif nums[mid]>target:
                    j=mid-1
                else:
                    i=mid+1
            return False
        start,end_row,end_col=0,len(matrix)-1,len(matrix[0])-1
        while(end_row>=start and end_col>=start):
            nums1=[ matrix[start][i] for i in range(start,end_col+1)]
            nums1+=[ matrix[i][end_col] for i in range(start,end_row+1)]
            nums2=[matrix[i][start] for i in range(start,end_row+1)]
            nums2+=[matrix[end_row][i] for i in range(start,end_col+1)]

            if binarysearch(nums1,target) or binarysearch(nums2,target):
                return True
            start+=1
            end_row-=1
            end_col-=1
        
        return False
So=Solution()
matrix=[[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
target=5
print(So.searchMatrix(matrix,target))
