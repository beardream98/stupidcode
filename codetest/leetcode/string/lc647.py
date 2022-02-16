class Solution:
    def mid_add(self,p,q,s):
        n=len(s)
        cnt=0
        while(p>=0 and q<n):
            if s[p]==s[q]:
                cnt+=1
                p,q=p-1,q+1
            else:
                break
        return cnt
    def countSubstrings(self, s: str) -> int:
        n=len(s)
        cnt=0
        for i in range(n-1):
            cnt+=self.mid_add(i,i,s)
            cnt+=self.mid_add(i,i+1,s)
        #  left_mid=n-1 right_mid=n-1
        cnt+=1
        return cnt
            
s=input("please enter a string:")
so=Solution()
print(so.countSubstrings(s))