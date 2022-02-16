from itertools import count


def countBits(n: int):
    count_list=[0]
    i,j,squre2=1,0,2
    while(i<=n):
        if i==squre2:
            j=0
            squre2=2*squre2
        count_list.append(count_list[j]+1)
        i+=1
        j+=1
    return count_list
n=eval(input("please enter a num:"))
print(countBits(n))



