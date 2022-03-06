from typing import List
from collections import defaultdict
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        q2v=defaultdict(list)
        visited={}
        for index,equation in enumerate(equations):
            
            q2v[equation[0]].append([equation[1],values[index]])
            q2v[equation[1]].append([equation[0],1/values[index]])
        visited={key:0 for key in q2v.keys()}

        def dfs(start,end,mul_num):
            if start==end:
                return mul_num
            visited[start]=1
            for v in q2v[start]:
                if visited[v[0]]!=1:
                    r_num=dfs(v[0],end,mul_num*v[1])
                    if r_num!=-1:
                        #多个query 要把visited 要把visited清除好 提前退出会遗漏清除
                        visited[start]=0
                        return r_num
            visited[start]=0
            return -1
        answer=[-1]*len(queries)
        for index,query in enumerate(queries):
            if  visited.get(query[0],-1)==-1 or  visited.get(query[1],-1)==-1:
                continue
            answer[index]=dfs(query[0],query[1],1)
        return answer
So=Solution()
equations=[["a","b"],["b","c"]]
values=[2.0,3.0]
queries=[["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
print(So.calcEquation(equations,values,queries))
              


