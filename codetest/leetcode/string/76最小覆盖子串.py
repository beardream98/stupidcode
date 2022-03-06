class Solution:
    def minWindow(self, s: str, t: str) -> str:
        save_str,min_len="",float("inf")
        save_dict={}
        for char in t:
            if save_dict.get(char,None)==None:
                save_dict[char]=1
            else:
                save_dict[char]+=1
        n,j=len(s),0
        for i,char in enumerate(s):
            if save_dict.get(char,None)==None:
                #当以i为首字符串，首字符不在跳过
                continue
            while(j<n and any([False if save_dict[key]<=0 else True for key in save_dict.keys()])):
                if save_dict.get(s[j],None)!=None:
                    save_dict[s[j]]-=1
                j+=1
            #保存字符
            if all([True if save_dict[key]<=0 else False for key in save_dict.keys()]):
                if (j-i)<min_len:
                    min_len=(j-i)
                    save_str=s[i:j]
            
            #去掉在i上的字符
            save_dict[char]+=1
        return save_str 
So=Solution()
s="cabwefgewcwaefgcf"
t="cae"
#边界测试
print(So.minWindow(s,t))