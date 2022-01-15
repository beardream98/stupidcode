def Paritition(people,low,high):
    
    key=people[low]
    while(low<high):
        ##此处 应该是<= 若存在相同值会进入互相抛的死循环，但是由于是两个值排序，等号写在第二层比较中
        while(low<high and (key[0]<people[high][0] or (key[0]==people[high][0] and key[1]>=people[high][1]))):
            high-=1
        people[low]=people[high]
        while(low<high and (key[0]>people[low][0] or (key[0]==people[low][0] and key[1]<=people[low][1]))):
            low+=1
        people[high]=people[low]
    people[low]=key
    return low

def quick_sort(people,low,high):
    if(low<high):
        mid=Paritition(people,low,high)
        quick_sort(people,low,mid-1)
        quick_sort(people,mid+1,high)
    return people


def reconstructQueue(people):
    low,high=0,len(people)-1
    quick_sort(people,low,high)
    if high<1:
        return people
    for j in range(high,-1,-1):
        cnt=people[j][1]
        temp_p=people[j]
        for i in range(cnt):
            people[j+i]=people[j+i+1]
        people[j+cnt]=temp_p
    return people



people=eval(input("please enter a list :"))
people = sorted(people, key = lambda x: (x[0], -x[1]))
print(people)
# print(reconstructQueue(people))