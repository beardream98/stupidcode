# 深搜与回溯
## 例题1 全排列
```python 
    def permute(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        
        visited=[0]*n
        return_list=[]
        def dfs(index,curr_list):
            if index==n:
                return_list.append(curr_list[:])
            
            for i in range(n):
                if visited[i]==0:
                    curr_list.append(nums[i])
                    visited[i]=1
                    dfs(index+1,curr_list)
                    visited[i]=0
                    curr_list.pop()
        dfs(0,[])
        return return_list
```
1. visited 的使用
   
   用来控制重复访问情况，防止死循环

2. 回溯使用

    当进入一次该节点本身存在多个选择时（而非多个往下方向），要使用回溯擦除本次搜索中前面的选择

3. [:]
   传递列表值时是用return_list保存下来的都是地址，深搜结束后会指向同一位置，要进行一次拷贝保存

## 同一层不选择相同内容
避免因同一层使用相同内容而出现的重复
```python
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        visited=[0]*n
        return_list=[]
        def dfs(index,curr_list):
            if index==n:
                return_list.append(curr_list[:])
            exist_dict=set()
            for i in range(n):
                if visited[i]==0 and nums[i] not in exist_dict:

                    exist_dict.add(nums[i])

                    curr_list.append(nums[i])
                    visited[i]=1
                    dfs(index+1,curr_list)
                    visited[i]=0
                    curr_list.pop()
        dfs(0,[])
        return return_list
```
1. set
   
   使用set在查询时速度更快，使用了hash表。set还可以通过update（可迭代对象）来进行更新

## a->b b->a 重复情况消除
``` python 

    def subsets(self, nums: List[int]) -> List[List[int]]:
        return_list=[]
        n=len(nums)
        def dfs(path,add_empty,begin):
            if add_empty:
                return_list.append(path[:])
                return 

            for i in range(begin,n):
                path.append(nums[i])
                dfs(path,False,i+1)
                path.pop()
            dfs(path,True,begin)
        dfs([],False,0)
        return return_list
```

1. a-b b-a 的重复情况

可以通过begin来进行排列组合，避免重复。当有重复元素时，注意排序避免b1-a
a-b2 的情况。排序后 a-b1-b2 

2. 叶子节点结束
如果需要在叶子节点进行递归出口，可以将其作为一个选择

## 树情况
与迭代列表不同，我们在每个index上进行选择。当为一棵树时，index->index+1类似，只不过选择不再是取值范围，而是哪个分叉节点。

```python
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        return_list=[]

        def dfs(Node,left_val,curr_list):
            if not Node.left and not Node.right:
                if left_val==Node.val:
                    return_list.append(curr_list[:]+[Node.val])
                return 
            left_val-=Node.val
            curr_list.append(Node.val)
            if Node.left:
                dfs(Node.left,left_val,curr_list)
            if Node.right:
                dfs(Node.right,left_val,curr_list)
            left_val+=Node.val

            curr_list.pop()
        if root:
            dfs(root,targetSum,[])
            return return_list

        else:
            return []
```
