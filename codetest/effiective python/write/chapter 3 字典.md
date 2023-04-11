# dict
1. 散列标准
   
   包含hash方法和qe方法。一般说来不可变对象都是可散列的，但如果一个tuple中如果含有不可散列元素则tuple不能散列

   自定义类型也是可散列，散列值为id返回值
2. 制造字典方法
   ```python
   >>> c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
   >>> d = dict([('two', 2), ('one', 1), ('three', 3)])
   >>> e = dict({'three': 3, 'one': 1, 'two': 2})
   ```
   将其作为包含了（key，value）元素的迭代器
3. 对dict.get的优化
   在更改某个键对应值时，使用get确定存在再改并不方便。
   ``` my_dict.setdefault(key,[]).append(new_value)```
   相当于查找时如果不存在自动将key和空列表放入映射中
4. defaultdict：
   类似的默认创建key 和在初始化时设置的类型放入映射中

5. update
   
   可以批量更新值，参数为可迭代对象
# set 和frozenset
1. 散列要求

   集合内部元素必须是可散列的，set本身不可以散列但是frozenset可以散列

2. 差、并、交
   通过对set进行操作，|得到并集，&得到交集 -得到差集
   ```python
   a=set([1,2,3])
   b=set([2,3,4])
   >>> a&b
   {2, 3}

   ```