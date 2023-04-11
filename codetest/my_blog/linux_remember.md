1、grep
grep [-type -data] filename
-C 上下多少行 -A 下方 -B 上方
2、单引号、双引号、不加引号
单引号 → 作为字符串
双引号=不加引号 → 会解析其中的命令
3、查看文件
**cat：**直接显示 -b显示行号
**more：**查看大文件，逐页显示
less：同more，但功能更强，可前后搜索，且less不会加载整个文件
向上搜=/ 向下搜=？ 重复搜索操作：n向下 N向上
**head：**查看开头 -n：前几行 -c：前多少字节
**tail：**查看末尾 -n：末几行 -c：末尾多少字节开始 -f：如果文件改变自动显示新增内容
4、>>与>
**>>：**追加 **>：**覆盖
5、<<与<
<<:从标准输入读入，遇到分界符停止 用法：命令 << 分界符
**<:**重定向，将后面跟的文件作为输入设备 用法：命令 < 文件
4与5结合：命令 < 文件1 > 文件2 解释：由文件1作为命令的输入，并将结果输出到文件2
例：cat << abc > file1 解释：以abc为终止符，从标准输入中读取数据存入file1中
6、cd
cd ~=cd
cd -：上次进入目录 cd ..：上一级 cd .：当前目录
7、ls
ls -a：包括隐藏目录
ls -l：详细信息
ll：ll=ls -al
8、pwd
当前目录
9、top 动态
查看系统内进程的资源占用情况
"-d time" 以time刷新状态
"-p pid" 专门监视某个pid
例如进程的pid、cpu占用、内存占用、运行时间、使用什么命令执行的
10、ps 静态
ps -aux
a：all (只包括当前终端的进程)

u：用户信息

x：其他终端的进程也显示
ps -elf
e：相当于ax

l：长格式，显示的信息多

f：完整的格式
-elf是标准格式 -aux是BSD(Berkeley Software Distribution)格式
11、sort
对结果排序，默认为第一列
"-k n" 按照第n列排序
"-r" 反序
12、touch
修改文件时间
"touch 文件" 如果没有该文件则创建，否则更改文件修改时间为当前时间
"touch -t date file" 修改文件时间戳
"touch -r A B" 修改文件B的时间戳与文件A时间戳相同
13、ifconfig
查看网卡信息
"ifconfig eth0 down/up" 关闭/启动网卡eth0
"ifconfig eth0 ip netmask 掩码" 在网卡eth0上设置某个ip的掩码
14、chmod
①chmod 777 file
三个7分别是文件拥有者、用户组、其他用户对文件的读写执行权限
7=二进制111=读-写-执行
②chmod u=rwx，g=rwx，o=rwx file
如果是chmod og= ，即后面啥也不跟，则清空权限
=可以为+或-，指增加/取消权限 例chmod a+r 即所有用户增加读权限
15、数据库
本地：mysql -u root -p
远程：mysql -h ip -u root -p
启动：
windows：net start mysql
linux：service mysqld start
16、mkdir
mkdir -p：可一次性创建多级目录
比如 mkdir -p a/b/c，如果a不存在的话会先创建a，再依次迭代创建b和c，不用手动cd进去创建
17、pushd、popd
目录栈，几个目录之间切换比较方便
18、软链接
ln -s 目标文件 软链接文件
19、alias
给命令设置别名
alias 别名='具体命令'
例如：alias open='xdg-open .'
注：设置后只有当前session有用，永久需要把命令添加到/etc/bash.bashrc
20、wc
用于查询文件的文本信息
-c 有多少字符
-l 多少行
-w 多少字符串
21、netstat
netstat -an | grep 端口号
a：所有socket，不管是不是监听
n：不显示ip、用户啥的别名，直接以数字显