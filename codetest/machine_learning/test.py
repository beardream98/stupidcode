from typing import List 
from collections import defaultdict
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        n, m = len(s), len(p)
        cnt = 0
        res = []
        count = defaultdict(int)
        if m > n or m <= 0:
            return 0
        
        for i in range(m):
        
            count[p[i]] -= 1

        for i in range(m):
            count[s[i]] += 1
            if count[s[i]] == 0:
                cnt += 1
        if m == cnt:
            res .append(0)

        for i in range(m,n):
            #去除
            if count[s[i - m]] == 0:
                cnt -= 1
            count[s[i - m]] -= 1

            if count[s[i]] == -1:
                cnt += 1
            count[s[i]] += 1
            
            if cnt == m:
                res.append(i - m + 1)
        return res 

s = "cbaebabacd"
p = "abc"
so = Solution()
res = so.findAnagrams(s,p)
print(res)
