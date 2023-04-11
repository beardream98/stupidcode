# -*- coding: utf-8 -*-
from email.errors import StartBoundaryNotFoundDefect
from pickle import TRUE
from re import I
from tkinter import ttk
import yaml
import data
from data.read_data import select_data,TrainData,file_split,DataByFile,DataByWord
from data.data_split import kfsplit
from data.datasets import prepare_loaders
from utils.wandb_init import get_run,wandb_utils
from utils.train import get_score,train_set,get_score_list
from utils.trivial import get_logger
from model.head import OriginModel,get_model
from model.trainer import train,predict
import torch
from transformers import logging
from data.read_data import read_data_file

import math
import wandb
import numpy as np
import pandas as pd 
import os
import torch

from sklearn.metrics import precision_score,recall_score,f1_score
import gc 
import torch.nn.functional as F
gc.enable()
from utils.trivial import set_seed
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import torch.distributed as dist
from data.datasets import make_dataloader,prepare_loaders
from utils.train import get_score,train_set,reduce_tensor,get_score_list,gather_tensor,gather_object
from utils.train import remove_module

from tqdm import tqdm

def predict(df,DEVICE,CONFIG,model_path=None,model=None):
    if model==None:
        CONFIG["Logger"].info(f"\nUsing{model_path}")
        model=get_model(CONFIG["model_name"])(CONFIG)
        model.to(DEVICE,non_blocking=True)
        if CONFIG["DIST_MODEL"]:
            args=CONFIG["args"]
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
                output_device=args.local_rank,find_unused_parameters=True)
            dist.barrier()
        ### may cause error 
            model.load_state_dict(torch.load(model_path))
        else:
            state_dict=remove_module(torch.load(model_path))
            model.load_state_dict(state_dict)
        
    model.eval()
    #set is_test=True to get label value
    data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],\
        CONFIG["MAX_LENGTH"],is_test=False,is_dist=CONFIG["DIST_MODEL"])

    predict_list,index_list=[],[]

    
    for batch in tqdm(data_loader):
        text_inputs=batch["text"]
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        index=batch["index"]
        with torch.no_grad():
            preds=model(**text_inputs)
            preds=F.softmax(preds,dim=1)
        predict_list+=preds.cpu().detach().numpy().tolist()
        index_list+=index.cpu().detach().numpy().tolist()
    
    assert(len(predict_list)==len(index_list))
    
    if CONFIG["DIST_MODEL"] :
        gather_array=np.concatenate([np.asarray(index_list).reshape(-1,1),predict_list],axis=1)
        gather_list=gather_object(gather_array,world_size=CONFIG["world_size"])
        index_array,predict_array=np.split(np.concatenate(gather_list,axis=0),[1],axis=1)
    else:
        index_array,predict_array=np.array(index_list),np.array(predict_list)

    
    predictions=np.argmax(predict_array,axis=1)
    '''
    多卡情况下，当数据不能被均等分配给各卡时，例如2卡，余1份数据，
    会将两份数据都分给两块卡，因此不用assert prediction与df大小
    '''
    # assert(len(predictions)==df.shape[0])

    index_array.astype(int)

    if index_array.ndim==2:
        index_array=np.squeeze(index_array)
    if predictions.ndim==2:
        predictions=np.squeeze(predictions)

    df["predict"]=-1
    
    df.loc[index_array,"prob0"]=predict_array[:,0]
    df.loc[index_array,"prob1"]=predict_array[:,1]

    df.loc[index_array,"predict"]=predictions

    if CONFIG["DIST_MODEL"]:
        dist.barrier()
    ## may cause error 
    gc.collect()
    return df

def filter_floop(CONFIG,trainDataObj,DEVICE):

    return 

def iter(select_word):
    
    file_path = os.path.join(CONFIG["INPUT_DIR"],select_word)
    positive_file = os.path.join(file_path,"UnlabelData")

    positive_df = read_data_file(positive_file,\
        CONFIG["DEBUG_MODEL"],data_num=CONFIG["DATA_NUM"],head=column_name)
    
    positive_df["fold"] = 1
    positive_df["label"] = positive_df["label"].replace(["0","1"],[0,1])
    positive_df = positive_df.query("label==1 or label==0")

    return positive_df

def get_binary_list(CONFIG):

    binary_word_path = os.path.join(os.path.dirname(CONFIG["INPUT_DIR"]),"classifyword.txt")
    binary_word_list = []
    with open(binary_word_path,"r") as fin:
        for line in fin:
            binary_word_list.append(line.strip())
    return binary_word_list

def get_swap_id(CONFIG):
    swap_id = []
    swap_id_path = os.path.join(os.path.dirname(CONFIG["INPUT_DIR"]),"swap_id")
    with open(swap_id_path,"r") as fin:
        for line in fin:
            swap_id.append(int(line.strip()))
    return swap_id

def get_unlabel_data(column_name,binary_word_list):
    data_df = pd.DataFrame(columns = column_name)
    for word in binary_word_list:
        temp_df = iter(select_word = word)
        data_df = pd.concat([data_df,temp_df],axis=0)
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    
    return data_df

def get_train_count(CONFIG,column_name,binary_word_list):
    CONFIG["PSEUDO_DATA"] = False
    train_data_count_path = os.path.join(CONFIG["INPUT_DIR"],"trainDataCount")
    if not os.path.exists(train_data_count_path):
        trainDataObj=DataByWord(CONFIG,"train",column_name=column_name)
        trainData=trainDataObj.get_alldata()
        train_word_num = {}
        for word in binary_word_list:
            positive_num = trainData.query("keyword == @word & label == 1")
            negative_num = trainData.query("keyword == @word & label == 0")
            train_word_num[word+"_positive"] = positive_num
            train_word_num[word+"_negative"] = negative_num
            
        torch.save(train_word_num,train_data_count_path)
    else:
        train_word_num = torch.load(train_data_count_path)

    return train_word_num

def persedo_write_process(CONFIG,DEVICE,swap_id,train_word_num,binary_word_list,data_df):
    min_num = min([value for value in train_word_num.values()])
    fold = 0 
    model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
    CONFIG["Logger"].info(f"\nUsing{model_path}")
    model=get_model(CONFIG["model_name"])(CONFIG)
    model.to(DEVICE,non_blocking=True)
    if CONFIG["DIST_MODEL"]:
        args=CONFIG["args"]
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
            output_device=args.local_rank,find_unused_parameters=True)
        dist.barrier()
    ### may cause error 
        model.load_state_dict(torch.load(model_path))
    else:
        state_dict=remove_module(torch.load(model_path))
        model.load_state_dict(state_dict)

    output_df = predict(data_df,DEVICE,CONFIG,model_path,model)

    unlabel_word_count = data_df["keyword"].value_counts().to_dict()
    min_num = min(unlabel_word_count.values())

    for word in tqdm(binary_word_list):
        word_file = os.path.join(CONFIG["INPUT_DIR"],word,"PseudoData")
        word_df = output_df.query("keyword == @word")
        positive_df = word_df.query("predict == 1")
        negative_df = word_df.query("predict == 0")

        positive_num,negative_num= train_word_num[word+"_positive"], train_word_num[word+"_negative"]
        min_binary_num = min(positive_num,negative_num)

        positive_sample_rate = ( math.sqrt(min_num/unlabel_word_count[word]) + math.sqrt(min_binary_num / train_word_num[word+"_positive"]) ) / 2 
        positive_sample_rate = positive_sample_rate ** CONFIG["SAMPLE_ALPHA"]

        negative_sample_rate = ( math.sqrt(min_num/unlabel_word_count[word]) + math.sqrt(min_binary_num/train_word_num[word+"_negative"]) ) / 2 
        negative_sample_rate = negative_sample_rate ** CONFIG["SAMPLE_ALPHA"]

        positive_sample_df = positive_df.sample(frac = positive_sample_rate)
        negative_sample_df = negative_df.sample(frac = negative_sample_rate)

        sample_df = pd.concat([positive_sample_df,negative_sample_df],axis = 0)
        sample_df["label"] = sample_df["predict"]
        if swap_id[binary_word_list.index(word)] == 1 :
            sample_df["label"].replace([0,1],[1,0])
        sample_df["label"] = sample_df["label"].replace([0,1],["0","1"])
        output_rows = sample_df[CONFIG["COLUMN_NAME"]].values.tolist()
        with open(word_file,"w") as write_object:
            for row in output_rows:
                write_object.write("\t".join(row)+"\n")

if __name__=='__main__':

    yaml_path="./config/train.yaml"
    CONFIG,Logger,DEVICE=train_set(yaml_path,experimentName=None,upload=False,filename="./test/logs",is_notebook = True)

    CONFIG["DEBUG_MODEL"]=False
    CONFIG["run_db"]=False
    CONFIG["SAVE_PATH"]="/root/autodl-tmp/epsave/checkpoint/best/binarybert"
    set_seed(CONFIG['SEED'])

    with open(os.path.join(CONFIG["INPUT_DIR"],"Schema"),"r") as f:
        line=f.readlines()[0].strip()
        column_name=line.split("\t")
        CONFIG["Logger"].info(f"column_name :{column_name}")
    CONFIG["COLUMN_NAME"]=column_name
    
    binary_word_list = get_binary_list(CONFIG)
    swap_id = get_swap_id(CONFIG)
    data_df = get_unlabel_data(column_name,binary_word_list)
    train_word_num = get_train_count(CONFIG,column_name,binary_word_list)

    persedo_write_process(CONFIG,DEVICE,swap_id,train_word_num,binary_word_list,data_df)

    