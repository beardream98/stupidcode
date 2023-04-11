# atx风格的标题
## 几个#代表几级标题
### 标题后跟一个空格 防止无法识别
*这是斜体*
**这是加粗**
***这是加粗倾斜***
* * * 
连续三个* 一行内代表分隔符
>这是一个引用,
>这仍然是一个引用

+ 列表标记1
- 还有-和*都可以作为列表标记 但是注意有一个空格
1. 而有序则通过1. 的方式来进行列表标记
* 列表也可以分级
  * tab可以形成分段

```python
for i in range(1):
    print("this is a code block")
```
`使用反引号包起来作为代码块`
    
    用一个空行代表换行

新的一行了
    

用两个空行代表空一行

|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |

|  表头1   | 表头2  | 表头3|
|  :----  | ----:  |:----:|
| 左对齐| 右对齐|居中对齐|

\* 通过转义符号\正确表示一些字符

内联方式插入公式 $a+b=2$ 需要使用两个\$符号并且无空格

独立显示公式

1. 公式使用
$${a \over b}=1$$
$\in$ 属于 $\notin$不属于

&emsp; 空格

pandoc --citeproc  --bibliography=myref.bib --csl=chinese-gb7714-2005-numeric.csl data_augment_paper.md -o data_augment_paper.docx

安装pip3 install pandoc-fignos

pandoc --filter pandoc-fignos --citeproc  --bibliography=myref.bib --csl=chinese-gb7714-2005-numeric.csl data_augment_paper.md -o data_augment_paper.docx

图片前后两行空开

{\over}-> frac{}{} 前者不能转换

pip install pandoc-eqnos --user 用于公式引用




pandoc --filter pandoc-eqnos --filter pandoc-fignos --citeproc  --bibliography=myref.bib --reference-doc=temple.doc data_augment_paper.md -o data_augment_paper.docxcd
