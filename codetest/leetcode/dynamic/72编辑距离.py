from re import L


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m,n=len(word1)+1,len(word2)+1
        dp=[ [0]*n for i in range(m)]
        for i in range(m):
            dp[i][0]=i
        for j in range(n):
            dp[0][j]=j
        for i in range(1,m):
            for j in range(1,n):
                deloperate_num=dp[i][j-1]+1
                insertoperate_num=dp[i-1][j]+1
                if word1[i-1]!=word2[j-1]:
                    changeoperate_num=dp[i-1][j-1]+1
                else:
                    changeoperate_num=dp[i-1][j-1]
                dp[i][j]=min(deloperate_num,insertoperate_num,changeoperate_num)
        return dp[m-1][n-1]


        

So=Solution()
word1=input("please enter a string :")
word2=input("please enter a string :")
print(So.minDistance(word1,word2))

                