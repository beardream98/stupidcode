# -*- coding: utf-8 -*-
from tkinter import E
import numpy as np
import pandas as pd 
import os
import random
import time
from tqdm import tqdm
import copy

from sklearn.model_selection import train_test_split
from .data_split import kfsplit

class TrainData(object):
    def __init__(self,df,CONFIG):
        self.data=df
        self.CONFIG=CONFIG
        positive_df,negative_df,negative_fold,dsat_df=self.data_encode()
        self.positive_df,self.negative_df,self.negative_fold,self.dsat_df=positive_df,negative_df,negative_fold,dsat_df
        
        self.iter_num=self.negative_fold
        self.it=self.iter()

    def get_data(self):
        return next(self.it)

    def iter(self):
        i=0
        while(1):
            sample_df=self.negative_df.query("sample_id==@i")
            i=(i+1)%self.iter_num
            sample_df.reset_index(inplace=True,drop=True)
            sample_df=kfsplit(sample_df,self.CONFIG,skip_info=True)

            train_df=pd.concat([self.positive_df,sample_df],axis=0)
            train_df=train_df.sample(frac=1).reset_index(drop=True)

            yield train_df

    def data_encode(self):
        df=self.data
        positive_df,negative_df=df[df["label"]==1].copy(),df[df["label"]==0].copy()
        dsat_num,dsat_df=0,None
        if self.CONFIG["KEEP_DSAT"]:
            dsat_df=negative_df[negative_df["ClickCount"]!="0"].copy()
            negative_df=negative_df[negative_df["ClickCount"]=="0"].copy()
            dsat_df.reset_index(inplace=True,drop=True)
            dsat_num=len(dsat_df)
            #encode dsat_df
            dsat_df["sample_id"]=-2
            dsat_df=kfsplit(dsat_df,self.CONFIG,skip_info=True)

        positive_df.reset_index(inplace=True,drop=True)
        positive_num=len(positive_df)
        negative_num=len(negative_df)
        
        # encode positive_df
        positive_df["sample_id"]=-1
        positive_df=kfsplit(positive_df,self.CONFIG,skip_info=True)
        

        # encode negative_df
        iter_num=positive_num-dsat_num
        sample_num=iter_num-(negative_num % iter_num)
        self.CONFIG["Logger"].info(f"negative_num:{negative_num},positive_num:{positive_num},add_num:{sample_num} \
            dsat_num:{dsat_num},iter_num:{iter_num}")

        sample_df=negative_df.sample(n=sample_num)
        negative_df=pd.concat([negative_df,sample_df],axis=0)
        negative_df.reset_index(inplace=True,drop=True)
        negative_df["sample_id"]=0

        for i in range(negative_num//iter_num +1):
            left,right=i*iter_num,(i+1)*iter_num
    
            negative_df.loc[left:right,"sample_id"]=i

        negative_fold=len(negative_df)/iter_num
        return positive_df,negative_df,negative_fold,dsat_df


def read_data_txt(input_file,schema_file,is_debug,data_num):
    with open(schema_file,"r") as input_object:
        schema_line=input_object.readlines()
    head=schema_line[0].strip().split("\t")
    with open(input_file,"r") as input_object:
        input_lines=input_object.readlines()
    if is_debug==True:
        input_lines=input_lines[:data_num]
    input_lines=[ line.strip().split("\t") for line in input_lines]
    input_lines=[ line for line in input_lines if len(line)==len(head)]
    data_rows=input_lines
    input_df=pd.DataFrame(data_rows,columns=head)
    
    return input_df
def sample_negative(df,negative_rate=1,seed=42):
    negative_df=df[df.label==0]
    positive_df=df[df.label==1]
    sample_num=len(positive_df)*negative_rate
    
    sample_df=negative_df.sample(n=sample_num,random_state=seed)
    
    # .sample(frac=1) 用于打乱顺序
    return_df=pd.concat([positive_df,sample_df]).sample(frac=1)
    return_df=return_df.reset_index(drop=True)
    return return_df

import langid
from tqdm._tqdm_notebook import tqdm_notebook
def language_classify(df):

    tqdm_notebook.pandas(desc='apply')
#     df["language"]=df.progress_apply(lambda r: langid.classify(r["Query"])[0], axis=1)
    df["language"]=df.Query.progress_apply(lambda r: langid.classify(r)[0])
    df_en=df[df["language"]=="en"]
        
    return df,df_en

def select_langugage(df,language="en"):

    tqdm_notebook.pandas(desc='apply')
    return_choose=df.Market.progress_apply(lambda r:r[0:2]==language)
    
    return_df=df[return_choose]
    print("filter rate:",len(return_df)/len(df))
    
    return return_df
# input_lang_df=language_classify(input_df)


def get_data(datapath):
    pos_files = os.listdir(datapath + '/pos')
    neg_files = os.listdir(datapath + '/neg')
    print(len(pos_files))
    print(len(neg_files))

    pos_all = []
    neg_all = []
    for pf, nf in zip(pos_files, neg_files):
        with open(datapath + '/pos' + '/' + pf, encoding='utf-8') as f:
            s = f.read()
            pos_all.append(s)
        with open(datapath + '/neg' + '/' + nf, encoding='utf-8') as f:
            s = f.read()
            neg_all.append(s)

    X_orig= np.array(pos_all + neg_all)
    Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])
    print("X_orig:", X_orig.shape)
    print("Y_orig:", Y_orig.shape)

    return X_orig, Y_orig
def select_data(CONFIG):
    if CONFIG["data_name"]=="imbd":
        x_orig,y_orig=get_data('/vc_data/users/v-mxiong/imdb/aclImdb/train')
        x_test,y_test=get_data('/vc_data/users/v-mxiong/imdb/aclImdb/test')
        data_df=pd.DataFrame({"Query":x_orig,"label":y_orig})
        data_df=data_df.sample(frac=0.1).reset_index(drop=True)

        test_df=pd.DataFrame({"Query":x_test,"label":y_test})
        test_df=test_df.sample(frac=0.1).reset_index(drop=True)
        CONFIG["Logger"].info("-"*10+"data_df label's value_counts"+"-"*10)
        CONFIG["Logger"].info(data_df["label"].value_counts())

    if CONFIG["data_name"]=="query":
        input_df=read_data_txt(CONFIG["INPUT_DIR"],CONFIG["SCHEMA_DIR"],CONFIG["DEBUG_MODEL"],CONFIG['DATA_NUM'])
    #     _,input_df=language_classify(input_df)
        if CONFIG["multiple_language"]==False: 
            input_df=select_langugage(input_df)
        
        input_df["label"]=input_df["label"].replace(["0","1"],[0,1])
        
        CONFIG["Logger"].info("-"*10+"input_df label's value_counts"+"-"*10)
        CONFIG["Logger"].info(input_df["label"].value_counts())

        data_df,test_df,y_train,y_test=train_test_split(input_df,input_df["label"],test_size=0.2,random_state=CONFIG["SEED"])
        test_df=sample_negative(test_df,1,CONFIG["SEED"])

        data_df.reset_index(inplace=True,drop=True)
        test_df.reset_index(inplace=True,drop=True)
        CONFIG["Logger"].info(f"size of data_df: {len(data_df)} , size of test_df: {len(test_df)}")
        CONFIG.update({"train size:":len(data_df)})    
    
    return data_df,test_df

## for big data
import math
def write_lines(lines,positive_path,negative_path,data_type):
    positive_num=0
    for line in lines:
        if line[1]=="1":
            positive_num+=1
    negative_fold=int(len(lines)/positive_num -1)+ math.ceil( (len(lines)%positive_num)/positive_num )
    positive_file=os.path.join(positive_path,data_type)
    negative_file=os.path.join(negative_path,data_type)
    
    neg_fold_id,negative_cnt=0,0
    positive_file_obj=open(positive_file,"w")
    negative_file_obj=open(negative_file+str(neg_fold_id),"w")

    for line in lines:
        if line[1]=="1":
            positive_file_obj.writelines("\t".join(line)+"\n")
        else:
            negative_cnt+=1
            negative_file_obj.writelines("\t".join(line)+"\n")
            if negative_cnt==positive_num:
                negative_file_obj.close()
                neg_fold_id+=1
                negative_file_obj=open(negative_file+str(neg_fold_id),"w")
                negative_cnt=0

    positive_file_obj.close()
    negative_file_obj.close()

    return negative_fold
def file_split(CONFIG,column_name=["Query","label"]):
    positivepath=os.path.join(CONFIG["INPUT_DIR"],"positivefold")
    negativepath=os.path.join(CONFIG["INPUT_DIR"],"negativefold")
    infopath=os.path.join(CONFIG["INPUT_DIR"],"info")

    if not os.path.exists(infopath):
        if not os.path.exists(positivepath):
            os.makedirs(positivepath)
        if not os.path.exists(negativepath):
            os.makedirs(negativepath)
        schema_file=os.path.join(CONFIG["INPUT_DIR"],"Schema")
        input_file=os.path.join(CONFIG["INPUT_DIR"],"LabelData")

        with open(schema_file,"r") as input_object:
            schema_line=input_object.readlines()
        head=schema_line[0].strip().split("\t")
        idxs=[ head.index(name) for name in column_name]
#         query_idx,label_idx,=head.index("Query"),head.index("label")

        with open(input_file,"r") as input_object:
            input_lines=input_object.readlines()
        input_lines=[ line.strip().split("\t") for line in input_lines]
        input_lines=[ [ line[idx] for idx in idxs] for line in input_lines if len(line)==len(head)]
        train_lines,test_lines,_,_=train_test_split(input_lines,[0]*len(input_lines),test_size=0.2,random_state=CONFIG["SEED"])

        train_negative_fold=write_lines(train_lines,positivepath,negativepath,"train")
        test_negative_fold=write_lines(test_lines,positivepath,negativepath,"test")

        with open(infopath,"w") as file_obj:
            file_obj.write(str(train_negative_fold)+"\t"+str(test_negative_fold))
    
    CONFIG["Logger"].info("file split completed")

def read_data_file(input_file,is_debug,data_num,head=["Query","label"]):
    with open(input_file,"r",encoding="UTF-8") as input_object:
        if is_debug==True:
            input_lines=[]
            for _ in range(data_num):
                input_lines.append(input_object.readline())
        else:
            input_lines=input_object.readlines()
    input_lines=[ line.strip().split("\t") for line in input_lines]
    input_lines=[ line for line in input_lines if len(line)==len(head)]

    input_df=pd.DataFrame(input_lines,columns=head)
    
    return input_df

class DataByFile(object):
    def __init__(self,CONFIG,data_type,column_name=["Query","label"],iter_now=-1):
        self.CONFIG=CONFIG
        self.positivepath=os.path.join(CONFIG["INPUT_DIR"],"positivefold")
        self.negativepath=os.path.join(CONFIG["INPUT_DIR"],"negativefold")
        self.data_type=data_type
        self.column_name=column_name
        # deperate the last uncomplete negative file
        self.iter_num=self.get_negative_fold()-1

        positive_file=os.path.join(self.positivepath,data_type)
        self.positive_df=self.get_fold(read_data_file(positive_file,\
            CONFIG["DEBUG_MODEL"],data_num=CONFIG["DATA_NUM"],head=self.column_name),skip_info=False)
        self.iter_now=iter_now
    def get_data(self,select_num=None):
        return self.iter(select_num)
    def check_data_multigpu(self,df):
        world_size=self.CONFIG["world_size"]
        if self.data_type=="train":
            batch_size=self.CONFIG["TRAIN_BATCH_SIZE"]*world_size
        else:
            batch_size=self.CONFIG["DEV_BATCH_SIZE"]*world_size
             
        if len(df)%batch_size<world_size and len(df)%world_size>0:
            extra_n=world_size-(len(df)%batch_size)
            extra_df=df.sample(n=extra_n,random_state=self.CONFIG["SEED"])
            df=pd.concat([df,extra_df],axis=0)
        return df
    def get_negative_fold(self):
        info_file=os.path.join(self.negativepath,"info")
        with open(info_file,"r",encoding="UTF-8") as file_obj:
            message=file_obj.readline().strip()
            message=message.split("\t")

        if self.data_type=="train":
            return int(message[0])
        else:
            return int(message[1])

    def get_fold(self,data_df,skip_info=True):

        data_df["label"]=data_df["label"].replace(["0","1"],[0,1])
        data_df=data_df.query("label==1 or label==0")
        data_df["label"]=data_df["label"].astype(int)
        data_df.reset_index(inplace=True,drop=True)
        data_df=kfsplit(data_df,self.CONFIG,skip_info=skip_info)
        return data_df


    def iter(self,select_num):
        if select_num!=None:
            i=select_num%self.iter_num
        else:
            self.iter_now+=1
            i=(self.iter_now)%self.iter_num
        negative_file=os.path.join(self.negativepath,self.data_type)

        negative_df=read_data_file(negative_file+str(i),\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
        self.replace_keyword(negative_df)
        negative_df=self.get_fold(negative_df)
        
        i=(i+1)%self.iter_num
        train_df=pd.concat([self.positive_df,negative_df],axis=0)
        if self.CONFIG["DIST_MODEL"]:
            train_df=self.check_data_multigpu(train_df)
            
        train_df=train_df.sample(frac=1).reset_index(drop=True)

        return train_df
        
class DataByFileHard(DataByFile):
    def __init__(self,CONFIG,data_type,column_name=["Query","label"],iter_now=-1,filter_rate=1,hard_rate=0.5):

        super(DataByFileHard,self).__init__(CONFIG,data_type,column_name,iter_now)
        
        self.hardNegativeFile=os.path.join(CONFIG["INPUT_DIR"],"hardnegfold",f"hardnegative_{filter_rate}")
        self.filter_rate=filter_rate
        self.hard_rate=hard_rate
        self.hardNegativeOffset=0

        # 当前fold
        self.fold=0
        self.hard_num=int(len(self.positive_df)*self.hard_rate*(CONFIG["NUM_FOLDS"]-1)/CONFIG["NUM_FOLDS"])
        
    def get_data(self,select_num=None):
        return self.iter(select_num)
        
    def readHardNegative(self,num):

        with open(self.hardNegativeFile,"r") as f:

            size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
            f.seek(self.hardNegativeOffset)
            lines = []
            for _ in range(num):
                if f.tell()> size:
                    #从头往里加
                    self.hardNegativeOffset=0
                    f.seek(0)
                line=f.readline()
                #只需要query部分即可
                line=line.strip().split("\t")[0]

                lines.append(line)

            self.hardNegativeOffset=f.tell()
    
        return lines

    def iter(self,select_num):

        if select_num!=None:
            i=select_num%self.iter_num
        else:
            self.iter_now+=1
            i=(self.iter_now)%self.iter_num
        negative_file=os.path.join(self.negativepath,self.data_type)
        negative_df=read_data_file(negative_file+str(i),\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
        negative_df=self.get_fold(negative_df)

        hard_query=self.readHardNegative(self.hard_num)
        
        replace_index=negative_df[negative_df["fold"]!=self.fold].sample(\
            n=self.hard_num,random_state=self.CONFIG["SEED"]).index
        negative_df.loc[replace_index,"Query"]=hard_query
        
        
        i=(i+1)%self.iter_num
        train_df=pd.concat([self.positive_df,negative_df],axis=0)
        if self.CONFIG["DIST_MODEL"]:
            train_df=self.check_data_multigpu(train_df)
            
        train_df=train_df.sample(frac=1).reset_index(drop=True)

        return train_df

class DataByWord(object):
    def __init__(self,CONFIG,data_type,column_name=["Query","label"],iter_now=-1):
        self.CONFIG = CONFIG
        self.data_type=data_type
        self.column_name=column_name
        
        binary_word_path = os.path.join(os.path.dirname(CONFIG["INPUT_DIR"]),"classifyword.txt")
        self.binary_word_list = []
        with open(binary_word_path,"r") as fin:
            for line in fin:
                self.binary_word_list.append(line.strip())
        self.swap_id = []
        swap_id_path = os.path.join(os.path.dirname(CONFIG["INPUT_DIR"]),"swap_id")
        with open(swap_id_path,"r") as fin:
            for line in fin:
                self.swap_id.append(int(line.strip()))
        self.iter_num=len(self.binary_word_list)
        self.iter_now=iter_now
        
    def get_data(self,select_word=None):
        return self.iter(select_word)

    def get_alldata(self):
        self.iter_now = -1 
        data_df = pd.DataFrame(columns=self.column_name)
        for _ in range(self.iter_num):
            temp_df = self.iter(select_word = None)
            data_df = pd.concat([data_df,temp_df],axis=0)
        data_df = data_df.sample(frac=1).reset_index(drop=True)

        return data_df 

    def check_data_multigpu(self,df):
        world_size=self.CONFIG["world_size"]
        if self.data_type=="train":
            batch_size=self.CONFIG["TRAIN_BATCH_SIZE"]*world_size
        else:
            batch_size=self.CONFIG["DEV_BATCH_SIZE"]*world_size
             
        if len(df)%batch_size<world_size and len(df)%world_size>0:
            extra_n=world_size-(len(df)%batch_size)
            extra_df=df.sample(n=extra_n,random_state=self.CONFIG["SEED"])
            df=pd.concat([df,extra_df],axis=0)
        return df

    def get_fold(self,data_df):

        data_df["label"] = data_df["label"].replace(["0","1"],[0,1])
        data_df = data_df.query("label==1 or label==0")
        data_df["label"] = data_df["label"].astype(int)
        data_df.reset_index(inplace=True,drop=True)
        
        #防止在验证集上的泄露，*10 是划分测试集的时候已经使用了%10,再用会有一个fold内数据为空
        data_df["fold"] = data_df["query_len"].astype(int) % (self.CONFIG["NUM_FOLDS"]+1) - 1
        return data_df

    def iter(self,select_word):
        if select_word != None:
            i = self.binary_word_list.index(select_word)
        else:
            self.iter_now += 1
            i = (self.iter_now)%self.iter_num

        # read train data
        file_path = os.path.join(self.CONFIG["INPUT_DIR"],self.binary_word_list[i])
        positive_file = os.path.join(file_path,"Positive"+self.data_type.capitalize())
        negative_file = os.path.join(file_path,"Negative"+self.data_type.capitalize())

        if self.CONFIG["DATA_AUG"] and self.data_type == "train" :
            positive_file = positive_file+"Aug"
        positive_df = read_data_file(positive_file,\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
        negative_df = read_data_file(negative_file,\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)

        if self.data_type == "train":
            # read val data
            positive_val_file = os.path.join(file_path,"Positive"+"Val")
            negative_val_file = os.path.join(file_path,"Negative"+"Val")
            positive_val_df = read_data_file(positive_val_file,\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
            negative_val_df = read_data_file(negative_val_file,\
            self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
            
            if self.CONFIG["PSEUDO_DATA"] == True:
                pseudo_file = os.path.join(file_path,"PseudoData")
                pseudo_df = read_data_file(pseudo_file,self.CONFIG["DEBUG_MODEL"],data_num=self.CONFIG["DATA_NUM"],head=self.column_name)
                positive_pseudo_df = pseudo_df.query("label == '1' ")
                negative_pseudo_df = pseudo_df.query("label == '0' ")
                positive_df = pd.concat([positive_df,positive_pseudo_df],axis=0)
                negative_df = pd.concat([negative_df,negative_pseudo_df],axis=0)

            positive_df["fold"] = 1
            positive_val_df["fold"] = 0
            negative_df["fold"] = 1 
            negative_val_df["fold"] = 0

            positive_df = pd.concat([positive_df,positive_val_df],axis=0)
            negative_df = pd.concat([negative_df,negative_val_df],axis=0)
        else:
            positive_df["fold"] = 1
            negative_df["fold"] = 1
 
        # 交换标签
        if self.swap_id[i] == 1:
            positive_df["label"] = 0
            negative_df["label"] = 1

        train_df = pd.concat([positive_df,negative_df],axis=0)
        if self.CONFIG["DIST_MODEL"]:
            train_df = self.check_data_multigpu(train_df)
        
        train_df["label"] = train_df["label"].replace(["0","1"],[0,1])
        train_df = train_df.query("label==1 or label==0")
        train_df["label"] = train_df["label"].astype(int)
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        return train_df

if __name__=='__main__':
    pass
