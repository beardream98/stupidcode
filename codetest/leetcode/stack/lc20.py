from typing import Mapping


def isValid(s):
    l1=[]
    dict1={"[":"]","(":")","{":"}","]":"",")":"","}":"",}
    for i,char in enumerate(s):
        if len(l1)==0 or char!=dict1[l1[-1]]:
            l1.append(char)
        else:
            l1.pop()
    if len(l1)==0:
        return True
    else:
        return False

s="[[[[]]]"    
print(isValid(s))


