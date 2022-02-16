class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        #p多于s的越界
        flag=False
        def bfs(s,p):
            nonlocal flag
            if s==p  :
                flag=True
                return
            elif len(s)==0 and len(p)!=0:
                if len(p)%2==0:
                    for i in range(len(p)/2):
                        if p[2*i]!="*":
                            return
                    flag=True
                return 
            elif len(p)==0 and len(s)!=0:
                return 
            
            s1,p1=s[0],p[0]
            if p1==s1 or p1==".":
                if p[1:2]!="*":
                    bfs(s[1:],p[1:])
                else:
                    # 为* 时三种情况 不进行匹配，匹配一个并继续
                    bfs(s,p[2:])
                    bfs(s[1:],p1+p[1:])
            if p1!=s1 and p[1:2]=="*":
                bfs(s,p[2:])
        bfs(s,p)

        return flag
so=Solution()
s=input("enter a string:")
p=input("enter a rule:")
print(so.isMatch(s,p))



