def decodeString(s):
    index=0
    s="1["+s+"]"
    n=len(s)
    def seg_cal():
        nonlocal index
        seg_str=""
        num=ord(s[index])-48
        index+=1
        while s[index]!="[":
            num*=10
            num+=ord(s[index])-48
            index=index+1
        index=index+1 
        while index<n:
            x=s[index]
            if x<="z" and x>="a":
                seg_str+=x
                index=index+1
            elif x<="9" and x>="0" :
                seg_str+=seg_cal() 
            elif x=="]": 
                index=index+1
                break
        return num*seg_str

    return seg_cal()
s="100[leetcode]"
print(decodeString(s))
