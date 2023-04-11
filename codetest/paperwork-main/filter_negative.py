# -*- coding: utf-8 -*-
from email.errors import StartBoundaryNotFoundDefect
from pickle import TRUE
from re import I
from tkinter import ttk
import yaml
import data
from data.read_data import select_data,TrainData,file_split,DataByFile
from data.data_split import kfsplit
from data.datasets import prepare_loaders
from utils.wandb_init import get_run,wandb_utils
from utils.train import get_score,train_set,get_score_list
from utils.trivial import get_logger
from model.head import OriginModel,get_model
from model.trainer import train,predict
import torch
from transformers import logging

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
    acc=(df["label"]==df["predict"]).mean()
    print(acc)
    gc.collect()
    return df
def write_negative_file(data_df,CONFIG,negative_path,threshold):

    hard_negative=data_df.query("label==0 & label!=predict ")

    for filter_rate in threshold:
        negative_file=os.path.join(negative_path,f"hardnegative_{str(filter_rate)}")
        write_lines=hard_negative.query(" prob1<=@filter_rate ")[CONFIG["Column_name"]].values.tolist()

        with open(negative_file,"a") as f:
            for line in tqdm(write_lines):
                line=list(map(str,line))
                f.writelines("\t".join(line)+"\n")


def filter_floop(CONFIG,trainDataObj,DEVICE,iter_max,start=-1,threshold=None):
    predict_test=[]
    for fold in range(CONFIG["NUM_FOLDS"]):
        if CONFIG["cross_val"]==False and fold>=1:
            break
        
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

        negative_path=os.path.join(CONFIG["INPUT_DIR"],"hardnegfold")

        if not os.path.exists(negative_path):
            os.makedirs(negative_path)
        for filter_rate in threshold:
            negative_file=os.path.join(negative_path,f"hardnegative_{str(filter_rate)}")
            with open(negative_file,"w"):
                # 清空文件夹
                pass 
        
        trainDataObj.iter_now=start
        for i in tqdm(range(start,iter_max)):
            df=trainDataObj.get_data()
            train_df=df[df["fold"]!=fold]
            train_df=predict(train_df,DEVICE,CONFIG,model_path,model)
            if CONFIG["args"].local_rank==0:
                write_negative_file(train_df,CONFIG,negative_path,threshold)
            if CONFIG["DIST_MODEL"]:
                dist.barrier()
    torch.cuda.empty_cache()
    gc.collect()

if __name__=='__main__':

    yaml_path="./config/train.yaml"
    CONFIG,Logger,DEVICE=train_set(yaml_path,experimentName=None,upload=False,filename="./test/logs")

    CONFIG["DATA_NUM"]=5000
    CONFIG["DEV_BATCH_SIZE"]=8192
    CONFIG["DEBUG_MODEL"]=False

    CONFIG["run_db"]=False
    ##todo 
    CONFIG["SAVE_PATH"]="/vc_data/users/v-mxiong/qc/checkpoint/best/qt13m"
    ##
    set_seed(CONFIG['SEED'])
    # file_split(CONFIG)
    CONFIG["Logger"].info("-"*10+"file_split finished already"+"-"*10)
    
    info_file=os.path.join(CONFIG["INPUT_DIR"],"negativefold","info")
    with open(info_file,"r",encoding="UTF-8") as file_obj:
        message=file_obj.readline().strip()
        message=message.split("\t")
    iter_max=int(message[0])
    
    with open(os.path.join(CONFIG["INPUT_DIR"],"Schema"),"r") as f:
        line=f.readlines()[0].strip()
        column_name=line.split("\t")
        CONFIG["Logger"].info(f"column_name :{column_name}")
    CONFIG["Column_name"]=column_name

    trainDataObj=DataByFile(CONFIG,"train",column_name=CONFIG["Column_name"])
    threshold=[1,0.95,0.9,0.85,0.8,0.75,0.7]
    filter_floop(CONFIG,trainDataObj,DEVICE,iter_max=iter_max,threshold=threshold)

   
