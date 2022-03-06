from typing import List
# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         flag=False
#         def dfs(s):
#             nonlocal flag
#             if s=="":
#                 flag=True
#                 return 
#             pre_s=""
#             for index,char in enumerate(s):
#                 pre_s+=char
#                 if wordDict.count(pre_s):
#                     dfs(s[index+1:])
#         dfs(s)
#         return flag

#记忆化搜索
# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         suffix_list=[]
#         def dfs(s):
#             if s=="":
#                 return True
#             pre_s=""
#             for index,char in enumerate(s):
#                 pre_s+=char
#                 if wordDict.count(pre_s) and not suffix_list.count(s[index+1:]):
#                     flag=dfs(s[index+1:])
#                     suffix_list.append(s[index+1:])
#                     if flag==True:
#                         return True
#             return False

#动态规划
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        T_index=[-1]
        for index,char in enumerate(s):
            for i in T_index:
                if ( wordDict.count(s[i+1:index+1])):
                    T_index.append(index)
                    break
        return (len(s)-1) in T_index


So=Solution()
s=input("please enter a string:")
wordDict=eval(input("please enter a list of str:"))
print(So.wordBreak(s,wordDict))

