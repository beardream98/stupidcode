def leastInterval( tasks,n):
    count_list=[0]*26
    task_num=len(tasks)
    for char in tasks:
        num=ord(char)-ord("A")
        count_list[num]+=1
    max_num=max(count_list)
    len_max=count_list.count(max_num)
    temp_time=(max_num-1)*(n+1)+len_max
    if temp_time<task_num:
        return task_num
    else:
        return temp_time

tasks=eval(input("please enter a list:"))
n=eval(input("please enter a number:"))
print(leastInterval(tasks,n))