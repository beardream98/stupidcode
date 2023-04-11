from collections import defaultdict
T=int(input())
for t in range(T):
    line1=list(map(lambda x:int(x),input().split()))
    n,k=line1[0],line1[1]
    s=input()
    num_s=[]
    dict_info=defaultdict(int)
    for char in s:
        num_s.append(ord(char))
        dict_info[ord(char)]+=1
    
    cnt,flag_index=0,n
    for i in reversed(range(n)):
        num=num_s[i]

        if (dict_info[num]+cnt)%k==0 :
            dict_info[num]+=cnt
            cnt=0
        elif dict_info[num]%k==0:
            continue
        else:
            cnt+=1
            dict_info[num]-=1
            flag_index=i
    if cnt!=0:
        print(-1)
    elif flag_index==n:
        print(s)
    else:
        output_s=s[:flag_index]
        for i in range(flag_index):
            dict_info[num_s[i]]-=1
        flag_num=num_s[flag_index]
        if [ item[0] for item in dict_info.items() if item[1]>0 and item[0]>=flag_num]==[]:
            print(-1)
        else:
            flag_num=min([ item[0] for item in dict_info.items() if item[1]>0 and item[0]>=flag_num])
            dict_info[flag_num]-=1
            rear_list=[]
            rear_list=sorted([ item[0]*item[1] for item in dict_info.items() if item[1]>0 ])
            output_s+=chr(flag_num)
            output_s+="".join([ chr(num) for num in rear_list])
            print(output_s)

        



# T=int(input())
# for t in range(T):
#     line1=list(map(lambda x:int(x),input().split()))
#     n,v=line1[0],line1[1]
#     ai,bi=[0]*n,[0]*n
#     for l in range(n):
#         line=list(map(lambda x:int(x),input().split()))
#         k,a,b=line[0],line[1],line[2]
#         ai[l],bi[l]=a,b
