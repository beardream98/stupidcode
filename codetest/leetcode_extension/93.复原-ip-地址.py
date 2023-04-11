#
# @lc app=leetcode.cn id=93 lang=python3
#
# [93] 复原 IP 地址
#
from typing import List

# @lc code=start
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        n=len(s)
        return_list=[]
        def dfs(index,depth,s_list):
            if index>=n or depth==4:
                if index==n and depth==4:
                    return_list.append(".".join(s_list))
                return 

            curr_s=""

            if int(s[index])==0:
                s_list.append("0")
                dfs(index+1,depth+1,s_list)
                s_list.pop()
            else:    
                for i in range(3):
                    if index+i>=n:
                        break
                    curr_s+=s[index+i]
                    if int(curr_s)>=0 and int(curr_s)<=255:
                        s_list.append(curr_s)
                        dfs(index+i+1,depth+1,s_list)
                        s_list.pop()
        dfs(0,0,[])
        return return_list

# @lc code=end
if __name__ =="__main__":
    so=Solution()
    s="25525511135"
    res=so.restoreIpAddresses(s)
    print(res)

