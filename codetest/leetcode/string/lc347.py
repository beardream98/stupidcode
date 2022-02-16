from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dict_count={}
        for num in nums:
            if num not in dict_count.keys():
                dict_count[num]=1
            else:
                dict_count[num]+=1
        cnt,maxk=0,[]
        for kv in sorted(dict_count.items(),key=lambda kv:(kv[1]),reverse=True):
            if cnt<k:
                maxk.append(kv[0])
                cnt+=1
            else:
                break
        return maxk

ss=Solution()
nums=eval(input("please enter a list:"))
k=eval(input("please enter k :"))
print(ss.topKFrequent(nums,k))



