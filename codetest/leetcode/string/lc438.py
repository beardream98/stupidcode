class Solution:
    def findAnagrams(self, s: str, p: str) :
        diff=[0]*26   
        index_list=[]     
        for char in p:
            diff[ord(char)-97]+=1
        m,n=len(s),len(p)
        if m<n:
            return []
        for i in range(n):
            diff[ord(s[i])-97]-=1
        if all([num==0 for num in diff ]):
            index_list.append(0)
        i,j=1,n
        while(j<m):
            diff[ord(s[i-1])-97]+=1
            diff[ord(s[j])-97]-=1
            if all([num==0 for num in diff ]):
                index_list.append(i)
            i+=1
            j+=1
        return index_list
solution1=Solution()
s=input("please enter a str:")
p=input("please enter a str:")
print(solution1.findAnagrams(s,p))




        