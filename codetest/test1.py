n=3
value=[1, 2, 3]
father_id=[0, 1, 1]

son_id=[[] for _ in range(n)]
for index,id in enumerate(father_id):
    if id==0:
        continue
    son_id[id-1].append(index)

ans=[v for v in value ]
def rec(father_id):
    if son_id[father_id]==[]:
        return ans[father_id]
    
    for id1 in son_id[father_id]:
        ans[father_id]+=rec(id1)
    
    return ans[father_id]

rec(0)
print(max(ans))

