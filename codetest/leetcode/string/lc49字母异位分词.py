from typing import List
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
        char2num={chr(i):prime[i-97] for i in range(97,123)}
        str2num_list=defaultdict(list)
        for str in strs:
            cnt=1
            for char in str:
                cnt*=char2num[char]
            str2num_list[cnt].append(str)

        return list(str2num_list.values())
so=Solution()
strs=eval(input("please enter a str list:"))
print(so.groupAnagrams(strs))

        