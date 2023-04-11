m,n,x=4,6,4
teacher_list=[["java","c++","python"],["python"],["c++","java"],["python"]]
student_list=["java","python","c++","python","c++","java"]
visit_map=[[0]*n for i in range(m)]
def hash_value(visit_map,m,n):
    cnt=0
    for i in range(m):
        for j in range(n):
            if visit_map[i][j]!=0:
                cnt+=i*n+j
    return cnt
# 用于记录失败的hash情况
mermory=[]
flag=False
def dfs(student_index,visit_map):
    global flag 
    if flag:
        return 
    if student_index==n :
        flag=True
        print("true")
        print(visit_map)
        return 
    #检查越界
    for i in range(m-1):
        for j in range(i+1,m):
            if (student_list[student_index] in teacher_list[i] and student_list[student_index] in teacher_list[j] 
                and sum(visit_map[i])<x and sum(visit_map[j])<x):
                visit_map[i][student_index]=1
                visit_map[j][student_index]=1
                hash=hash_value(visit_map,m,n)
                if hash not in mermory:
                    mermory.append(hash)
                    dfs(student_index+1,visit_map)
                visit_map[i][student_index]=0
                visit_map[j][student_index]=0
dfs(0,visit_map)
if not flag:
    print("false")
                


