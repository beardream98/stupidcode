#
# @lc app=leetcode.cn id=3 lang=python3
#
# [3] 无重复字符的最长子串
#

# @lc code=start


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n=len(s)
        char_dict={}
        uni_num,max_uni_num,i=0,0,0
        for j in range(n):
            while(i<j and char_dict.get(s[j],0)==1):
                uni_num-=1
                char_dict[s[i]]=0
                i+=1
            if char_dict.get(s[j],0)==0:
                uni_num+=1
                char_dict[s[j]]=1
            max_uni_num=max(uni_num,max_uni_num)
        return max_uni_num
# @lc code=end
with open("testwrite","r") as fileobject:
    s=fileobject.readline().strip()
n=len(s)
char_dict={}
uni_num,max_uni_num,i=0,0,0
for j in range(n):
    while(i<j and char_dict.get(s[j],0)==1):
        uni_num-=1
        char_dict[s[i]]=0
        i+=1
    if char_dict.get(s[j],0)==0:
        uni_num+=1
        char_dict[s[j]]=1
    max_uni_num=max(uni_num,max_uni_num)
    
