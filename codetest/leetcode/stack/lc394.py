#递归解法
def decodeString(s):
    s="1["+s+"]"
    def decode_one(index):
        num_fold=""
        while(ord(s[index])<=57 and ord(s[index])>=48):
            num_fold+=s[index]
            index+=1
        num_fold=int(num_fold)
        #跳过[
        index+=1
        temp_s=""
        while(s[index]!="]"):
            if ord(s[index])<=122 and ord(s[index])>=65:
                temp_s+=s[index]
                index+=1
            else:
                temp_s_next,index=decode_one(index)
                temp_s+=temp_s_next
                index+=1
        temp_s=temp_s*num_fold
        return temp_s,index
    decodes,index=decode_one(0)
    return decodes

s="10[a]"
print(decodeString(s))
    