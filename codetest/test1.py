import heapq
from collections import Counter
from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        heap = []
        for key,value in count.items():
            if len(heap) >= k:
                if value > heap[0][0]:
                    heapq.heapreplace(heap,(value,key))
            else:
                heapq.heappush(heap,(value,key))
        print(heap)
        return [ t[1] for t in heap ]
    
s = Solution()
nums = [1,1,1,2,2,3]
k = 2
res = s.topKFrequent(nums,k)
print(res)

