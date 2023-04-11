# 背包九讲
## 01 背包问题
题目：有N件物品和一个容量为V的背包。放入第i将物品耗费的费用是$c_i$,得到的价值是$w_i$。求解如何使价值最大

1. 状态方程
   $F[i,v]=max\{F[i-1,v],F[i-1,v-c_i]+w_i\}$
   F[i,v]表示的是在选择i件物品后总价值最大。转移过程用是否选择第i件物品来决定。

2. 伪代码
   
   ```python
    dp=[ [0]*N for _ in range(V+1)]
    
    for i in range(N):
        for j in range(c_i,V+1):
            dp[i,j]=max(dp[i-1,j],dp[i-1,j-c[i]]+w[i])
    
   ```
   > 取v+1是假设存在一个容量为0的状态，方便初始化以及正确赋值
3. 优化思路

    不需要存储每一行，只需保存上一行即可。由状态转移方程，
    $F[i,v]=max{F[i-1,v],F[i-1,v-c_i]+w_i}$，通过从后往前遍历v可以直接在原数组上进行修改。
   ```python
    dp=[0]*(V+1)
    for i in range(N):
        for j in reversed(range(c_i,V)):
            dp[j]=max(dp[j],dp[j-c[i]]+w[i])
   ```
   >注意：优化思路中只对c_i后开始操作，基本假设就是对于小于c_i的直接取上一轮迭代的结果

4. 是否装满

    两种问法，1是装满 2是不要求最大值即可。转移方程不受影响，初始化受到影响。
    ```python
        dp[0]=[float(-inf)]*(V+1) 
        dp[0]=0
    ```
    即在背包为空的情况下，只有容量为0才是合法解。

## 完全背包问题
当问题转换为完全背包，即每件物品可以取无限件的时候。
1. 状态转移方程
   $F[i,v]=max{F[i-1,v],F[i,v-c_i]+w_i}$
    决策由是否取第i件变为是否该再取一件
2. 方程合理性
   
   不妨假设当$v>=c_i \& v<2*c_i$,此时有$F[i,v-c_i]=F[i-1,v-c_i]$,由于$v-c_i<c_i$在i位置选不了，故相等。方程合理
   
   当$v>=2*c_i \& v<3*c_i$分为两种情况，是否取一件为$c_i$的物品和是否取一件为$2*c_i$的物品。则方程为:
   $$
   max\{F[i-1,v],F[i-1,v-c_i]+w_i\} \tag{1}
   $$
   $$
   max\{F[i-1,v],F[i-1,v-2*c_i]+2*w_i\} \tag{2}
   $$
   综上两种情况，我们的结果为
   $$
   F[i,v]=max\{F[i-1,v],F[i-1,v-2*c_i]+2*w_i,\\
   F[i-1,v-c_i]+w_i\}   \tag{3}
   $$
   对于$F[i,v-c_i]$我们有
   $$
   F[i,v-c_i]=max\{F[i-1,v-c_i],F[i-1,v-2*c_i]+w_i\} \tag{4}
   $$
   在等式两边加上一个$w_i$刚好有
   $$
   F[i,v-c_i]+w_i=max\{F[i-1,v-c_i]+w_i,\\
   F[i-1,v-2*c_i] +2*w_i\}  \tag{5}
   $$
   即为$(3)$中后两项，因此有$F[i,v]=max{F[i-1,v],F[i,v-c_i]+w_i}$，对于v大于$3*c_i$的情况可以依次类推。

# leetcode实例
## 0/1背包问题
### 416
&emsp;&emsp;给你一个只包含正整数的非空数组nums。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

&emsp;&emsp;转换为0/1背包问题，分隔两个子集问题转换成选取数个数字使得值为和的一半。此时背包容量变为和的一半，值为能否做到。`dp[j]=max(dp[j-num],dp[j])`

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target=sum(nums)
        n=len(nums)
        if target%2!=0 or n==1:
            return False
        dp=(target+1)*[0]
        dp[0]=1

        for num in nums:
            for j in reversed(range(num,target+1)):
                dp[j]=max(dp[j-num],dp[j])
        return dp[target//2]==1
```

### 494
&emsp;&emsp;给你一个整数数组nums和一整数target。向数组中的每个整数前添加'+'或'-' ，然后串联起所有整数，可以构造一个表达式 ：

&emsp;&emsp;例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。返回可以通过上述方法构造的、运算结果等于target的不同表达式的数目。


构造```target=(sum-neg)-neg neg=(sum-target)/2```据此去选取数字使得值为neg

```python
from collections import defaultdict

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # target=(sum-neg)-neg
        # neg=(sum-target)/2
        neg=sum(nums)-target
        if neg<0 or neg%2!=0:
            return 0
        neg=neg//2
        dp=[0]*(neg+1)
        dp[0]=1
        for i in range(len(nums)):
            for j in reversed(range(nums[i],neg+1)):
                dp[j]=dp[j-nums[i]]+dp[j]
        return dp[neg]
```

## 完全背包问题
### 518
&emsp;&emsp;给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

&emsp;&emsp;请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。假设每一种面额的硬币有无限个。

典型的完全背包问题

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp=[0]*(amount+1)
        dp[0]=1
        for coin in coins:
            for j in range(coin,amount+1):
                dp[j]+=dp[j-coin]
        
        return dp[amount]
```
## 二维背包问题
### 474 
给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

构建二维的背包，其余和0/1背包一致

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        #将物品价值设为1，转换为0/1背包问题。（多了一维）
        dp=[(n+1)*[0] for _ in range(m+1)]
        #需要空的状态

        for str in strs:
            zero_num=str.count("0")
            one_num=str.count("1")
            for i in reversed(range(zero_num,m+1)):
                for j in reversed(range(one_num,n+1)):
                    dp[i][j]=max(dp[i][j],dp[i-zero_num][j-one_num]+1)
        return dp[m][n]
```