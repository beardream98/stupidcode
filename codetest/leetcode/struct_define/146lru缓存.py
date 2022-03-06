class LRUCache:

    def __init__(self, capacity: int):
        self.capacity=capacity
        self.cur_cap=0
        self.lrudict={}
        self.userank=[]
        self.rankcount={}

    def get(self, key: int) -> int:
        if self.lrudict.get(key,-1)!=-1:
            #能找到的情况下更新使用情况
            self.userank.append(key)
            self.rankcount[key]+=1
        return self.lrudict.get(key,-1)

    def put(self, key: int, value: int) -> None:
        if self.lrudict.get(key,-1)!=-1:
            self.lrudict[key]=value
            self.userank.append(key)
            self.rankcount[key]+=1
            return 
        elif self.cur_cap==self.capacity:
            #逐出一个
            while(self.rankcount[self.userank[0]]>1):
                self.rankcount[self.userank[0]]-=1
                self.userank.pop(0)
            del self.lrudict[self.userank[0]]
            self.rankcount[self.userank[0]]=0
            self.userank.pop(0)
            self.cur_cap-=1
        
        #此时容量足够插入一个
        self.lrudict[key]=value
        self.userank.append(key)
        self.rankcount[key]=1
        self.cur_cap+=1
capacity=2
obj = LRUCache(capacity)
obj.put(2,1)
obj.put(1,1)
obj.put(2,3)
obj.put(4,1)
print(obj.get(1),obj.get(2))

