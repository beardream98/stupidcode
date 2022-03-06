from typing import List

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        s="*"+s
        n=len(s)
        string_list=[]
        string_len=0
        def dfs(index,string,left_num,choose):
            nonlocal n,string_list,string_len
            if index==n  :
                if left_num==0 and len(string)>string_len:
                    string_list=[]
                    string_list.append(string)
                    string_len=len(string)
                elif left_num==0 and len(string)==string_len:
                    string_list.append(string)
                return 
            #进入递归 选择当前字符串
            if choose:
                if s[index]=="(":
                    left_num+=1
                elif s[index]==")":
                    left_num-=1
                string+=s[index]
                # 不合法直接跳出
                if left_num<0:
                    return 
                
            # 选择
            dfs(index+1,string,left_num,True)
            # 当为字母时 只有选择这一种情况
            if index+1<n and s[index+1]!="(" and s[index+1]!=")":
                return 
            
            dfs(index+1,string,left_num,False)
            

        # 重复问题
        #贪心最大化问题
        dfs(1,"",0,True)
        return list(set(string_list))

        
s="()())()"
So=Solution()
print(So.removeInvalidParentheses(s))


