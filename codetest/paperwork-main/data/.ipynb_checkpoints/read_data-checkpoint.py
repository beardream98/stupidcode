# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import os
import random
import time
from tqdm import tqdm
import copy

from sklearn.model_selection import train_test_split



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
        
        input_df=sample_negative(input_df,1,CONFIG["SEED"])
        CONFIG["Logger"].info("-"*10+"input_df label's value_counts"+"-"*10)
        CONFIG["Logger"].info(input_df["label"].value_counts())

        data_df,test_df,y_train,y_test=train_test_split(input_df,input_df["label"],test_size=0.2,random_state=CONFIG["SEED"])
        data_df.reset_index(inplace=True,drop=True)
        test_df.reset_index(inplace=True,drop=True)
        CONFIG["Logger"].info(f"size of data_df: {len(data_df)} , size of test_df: {len(test_df)}")
        CONFIG.update({"train size:":len(data_df)})    
    
    return data_df,test_df


if __name__=='__main__':
    pass
