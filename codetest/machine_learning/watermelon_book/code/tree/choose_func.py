from math import log
from classes import dataset_sub

def Ent(D):
    information_entropy=0
    for label in D.num_dict.keys():
        
        p=D.num_dict[label]/len(D)
        if p==0:
            continue
        information_entropy-=p*log(p,2)
    return information_entropy
def Gain(D,a):
    # a 代表属性也是x中某个维度 
    gain_d=Ent(D)
    for value in a.values:
        d=dataset_sub(D,a.dimension,value)
        gain_d-=len(d)/len(D)*Ent(d)

    return gain_d



if __name__=="__main__":
    pass

