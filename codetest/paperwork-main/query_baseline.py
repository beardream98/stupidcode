# -*- coding: utf-8 -*-
import yaml
import data
from data.read_data import select_data,TrainData,file_split,DataByFile,DataByFileHard,DataByWord
from data.data_split import kfsplit
from data.datasets import prepare_loaders
from utils.wandb_init import get_run,wandb_utils
from utils.train import get_score,train_set,get_score_list
from utils.trivial import get_logger
from model.head import OriginModel,LR,get_model
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
gc.enable()
from utils.trivial import set_seed

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def train_floop(CONFIG,data_df,DEVICE,trainDataObj=None):
    if not os.path.exists(CONFIG["SAVE_PATH"]):
        os.makedirs(CONFIG["SAVE_PATH"])
    for fold in range(CONFIG["NUM_FOLDS"]):

        if CONFIG["cross_val"]==False and fold>=1:
            break
        CONFIG["Logger"].info(f"----------fold {fold}----------")
        run=get_run(CONFIG,"Train",fold)
        
        model=get_model(CONFIG["model_name"])(CONFIG)
        
        train(model,data_df,fold,CONFIG,DEVICE,run=run,trainDataObj=trainDataObj)
        
        # _,_,acc_train_lastepoch=predict(data_df,DEVICE,CONFIG,model_path=None,model=model)
        
        # CONFIG["Logger"].info(f"train{fold} acc in last epoch:{acc_train_lastepoch}")
        del model
        torch.cuda.empty_cache()
        if run!=None:
            run.finish()

    gc.collect()

def dev_floop(CONFIG,data_df,DEVICE):
    #no 交叉验证 many error 
    prediction_dev=np.zeros(len(data_df))
    run=get_run(CONFIG,"dev",0)

    for fold in range(CONFIG["NUM_FOLDS"]):

        # 不用交叉验证
        if CONFIG["cross_val"]==False and fold>=1:
            break
        model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
        

        train_df,val_df=data_df[data_df.fold!=fold],data_df[data_df.fold ==fold]
        val_index=val_df.index
        _,_,acc_train,_=predict(train_df,DEVICE,CONFIG,model_path,None)
        _,predictions,acc_val,val_labels=predict(val_df,DEVICE,CONFIG,model_path)
        
        CONFIG["Logger"].info(f"----------fold {fold} dev metric----------")
        CONFIG["Logger"].info(f"val-fold{fold}:  acc train:{acc_train}")
        CONFIG["Logger"].info(f"val-fold{fold}:  acc val:{acc_val}")

        precesion,recall,f1=get_score_list(val_labels,predictions,CONFIG["average"])
        CONFIG["Logger"].info(f"val-fold{fold}:  precesion:{precesion},recall:{recall},f1:{f1}")
        prediction_dev[val_index]=predictions

    if CONFIG["cross_val"]==True:
        acc_val=(data_df["label"]==prediction_dev).mean()
        precesion,recall,f1=get_score(data_df,predictions,CONFIG["average"])
        CONFIG["Logger"].info(f"----------summary dev metric----------")
        CONFIG["Logger"].info(f"acc_val:{acc_val} precesion:{precesion},recall:{recall},f1:{f1}")
        
    if run!=None:
        run.summary["val-acc"]=acc_val
        run.summary["val-precesion"]=precesion
        run.summary["val-recall"]=recall
        run.summary["val-f1"]=f1
        run.finish()
    torch.cuda.empty_cache()
    gc.collect()


def test_floop(CONFIG,test_df,DEVICE):
    predict_test=[]
    for fold in range(CONFIG["NUM_FOLDS"]):
        if CONFIG["cross_val"]==False and fold>=1:
            break
        model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
        test_df,acc_test=predict(test_df,DEVICE,CONFIG,model_path)
        predict_list,predictions,test_labels=test_df[["prob0","prob1"]].values.tolist()\
            ,test_df["predict"].values.tolist(),test_df["label"].values.tolist()

        if predict_test==[]:
            predict_test=np.asarray(predict_list)
        else:
            predict_test+=np.asarray(predict_list)
    
    run=get_run(CONFIG,"test",0)

    prediction_test=np.argmax(predict_test,axis=1)
    acc=(np.asarray(test_labels)==prediction_test).mean()

    precesion,recall,f1=get_score_list(test_labels,predictions,CONFIG["average"],position=0)
    CONFIG["Logger"].info(f"----------summary test metric----------")
    CONFIG["Logger"].info(f"label0:test-acc:{acc} test:precesion:{precesion},recall:{recall},f1:{f1}")
    
    precesion,recall,f1=get_score_list(test_labels,predictions,CONFIG["average"])
    CONFIG["Logger"].info(f"test-acc:{acc} test:precesion:{precesion},recall:{recall},f1:{f1}")
    if run!=None:
        run.summary["test-acc"]=acc
        run.summary["test-precesion"]=precesion
        run.summary["test-recall"]=recall
        run.summary["test-f1"]=f1
        run.finish()
    torch.cuda.empty_cache()
    gc.collect()
    return test_df
if __name__=='__main__':

    yaml_path="./config/train.yaml"
    CONFIG,Logger,DEVICE=train_set(yaml_path,experimentName=None,upload=False)

    set_seed(CONFIG['SEED'])

    # file_split(CONFIG)
    CONFIG["Logger"].info("-"*10+"file_split finished already"+"-"*10)
    with open(os.path.join(CONFIG["INPUT_DIR"],"Schema"),"r") as f:
        line=f.readlines()[0].strip()
        column_name=line.split("\t")
        CONFIG["Logger"].info(f"column_name :{column_name}")
        CONFIG["COLUMN_NAME"]=column_name
        
    if CONFIG["HARD_NEG"]:
        filter_rate,hard_rate=CONFIG["FILTER_RATE"],CONFIG["HARD_RATE"]
        CONFIG["Logger"].info("-"*10+f"filter_rate: {filter_rate} hard_rate: {hard_rate}"+"-"*10)
        trainDataObj=DataByWord(CONFIG,"train",column_name=column_name,filter_rate=filter_rate,hard_rate=hard_rate)
    else:
        trainDataObj=DataByWord(CONFIG,"train",column_name=column_name)

    
    trainData=trainDataObj.get_alldata()
    CONFIG["Logger"].info("-"*10+"trainData label's value_counts"+"-"*10)
    CONFIG["Logger"].info(trainData["label"].value_counts())
    CONFIG["Logger"].info("-"*10+"trainData fold's value_counts"+"-"*10)
    CONFIG["Logger"].info(trainData["fold"].value_counts())


    testDataObj=DataByWord(CONFIG,"test",column_name=column_name)

    testData=testDataObj.get_alldata()

    if not CONFIG["different_negative"]:
        trainDataObj=None
    train_floop(CONFIG,trainData,DEVICE,trainDataObj=trainDataObj)
    ######## dev_floop(CONFIG,trainData,DEVICE)
    test_floop(CONFIG,testData,DEVICE)

   
    