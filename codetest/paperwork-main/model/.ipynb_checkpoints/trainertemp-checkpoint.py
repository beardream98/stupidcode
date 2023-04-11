# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import os
import random
import time
from tqdm import tqdm
import copy


import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, logging

from .tricks import EarlyStopping,get_parameters
from data.datasets import make_dataloader,prepare_loaders
from .head import OriginModel
from utils.train import get_score,train_set
from data.read_data_temp import TrainData
import gc 
gc.enable()


def evaluate(model,dev_dataloader,DEVICE):
    model.eval()
    
    total_val_loss=0
    for batch in dev_dataloader:
        text_inputs=batch["text"]
        label=batch["label"].to(DEVICE)
        
        text_inputs={key: value.to(DEVICE) for key,value in text_inputs.items()}
        text_inputs["labels"]=label
        with torch.no_grad():
            loss=model(**text_inputs)
            
        total_val_loss+=loss.item()
        
    return total_val_loss/len(dev_dataloader)
def train(model,data_df,fold,CONFIG,DEVICE,run=None,trainDataObj=None):
    
    
    train_df,val_df=data_df[data_df.fold!=fold],data_df[data_df.fold==fold]
    train_dataloader,dev_dataloader=prepare_loaders(data_df,fold,CONFIG)

#     optimizer=AdamW(get_parameters(model, model_init_LR=CONFIG["LR"]*1.5, multiplier=0.975, classifier_LR=CONFIG["LR"]), LR = CONFIG["LR"], EPS= CONFIG["EPS"])
    optimizer=AdamW(model.parameters(), lr = CONFIG["LR"])
    """
    get_linear_schedule_with_warmup:学习率先从0开始warm_up到设定学习率，再逐渐减到0
    num_warmup_steps：完成预热的步数
    num_training_steps：训练批次*CONFIG["EPOCHS"] 训练的step数
    """

    num_warm_rate=CONFIG["NUM_WARM_RATE"]
    num_training_steps=len(train_dataloader)*CONFIG["EPOCHS"]
    num_warm_steps=int(num_warm_rate*num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warm_steps, 
                                                num_training_steps=len(train_dataloader) * CONFIG["EPOCHS"])
    scaler = GradScaler(enabled=CONFIG["FP16"])
    
    best_val_loss=100
    best_val_acc=-1
    best_val_f1=-1
    best_model_dict=None
    
    es=EarlyStopping(patience=CONFIG["early_patience"])
    
    if CONFIG["run_db"]==True:
        wandb.watch(model,log_freq=100)
    
    CONFIG["Logger"].info("-"*10+"train start"+"-"*10)

    for epoch in range(CONFIG["EPOCHS"]):
        sum_train_loss=0
        start = time.time()
        model.train()
        if epoch!=0 and trainDataObj!=None:
            data_df=trainDataObj.get_data()
            train_df,val_df=data_df[data_df.fold!=fold],data_df[data_df.fold==fold]
            train_dataloader,dev_dataloader=prepare_loaders(data_df,fold,CONFIG)
        for index,batch in enumerate(tqdm(train_dataloader,leave=False)):
            
            model.zero_grad()
            text_inputs=batch["text"]
            label=batch["label"].to(DEVICE)
        
            text_inputs={key: value.to(DEVICE) for key,value in text_inputs.items()}
            text_inputs["labels"]=label
            with autocast(enabled=CONFIG["FP16"]):
                loss=model(**text_inputs)
            sum_train_loss+=loss.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

#             loss.backward()
#             optimizer.step()
            scheduler.step()
            
        avg_train_loss=sum_train_loss/len(train_dataloader)
        val_loss=evaluate(model,dev_dataloader,DEVICE)
        _,predictions,val_acc=predict(val_df,DEVICE,CONFIG,None,model)
        precesion,recall,f1=get_score(val_df,predictions,CONFIG["average"],position=1)

        CONFIG["Logger"].info(f"EPOCH:{epoch+1}/{CONFIG['EPOCHS']} train_loss:{avg_train_loss},val_loss:{val_loss},\
            f1:{f1}")
        CONFIG["Logger"].info(f"EPOCH:{epoch+1}/{CONFIG['EPOCHS']} dev_acc:{val_acc}")
        
        time_used = time.time()-start
        CONFIG["Logger"].info(f"one_epoch_time:{time_used}")
        
        if val_acc>best_val_acc:
            best_val_acc=val_acc

        if val_loss<best_val_loss:
            best_val_loss=val_loss

        if f1>best_val_f1:
            best_val_f1=f1
            best_model_dict=copy.deepcopy(model.state_dict())
            CONFIG["Logger"].info(f"best_model saved ,f1:{best_val_f1},recall:{recall},val acc:{val_acc}")

        if CONFIG["run_db"]==True:
            wandb.log({"Train LOSS":avg_train_loss,"Valid LOSS":val_loss})
            wandb.log({"Valid ACC":val_acc})
            wandb.log({"Valid F1":f1})
            run.summary["best_val_loss"]=best_val_loss
            run.summary["best_val_acc"]=best_val_acc
            run.summary["best_val_f1"]=best_val_f1
            
        if es.step(val_loss):
            break
    CONFIG["Logger"].info("-"*10+"train finished"+"-"*10)
    
    CONFIG["Logger"].info(f"best_val_acc:{best_val_acc},best_val_loss:{best_val_loss},best_val_f1:{best_val_f1}")
    return best_val_loss,best_model_dict

def predict(df,DEVICE,CONFIG,model_path=None,model=None):
    if model==None:
        CONFIG["Logger"].info(f"\nUsing{model_path}")
        model=OriginModel(CONFIG)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        
    model.eval()
    
    data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],CONFIG["MAX_LENGTH"],is_test=True)
    predict_list=[]
    for batch in data_loader:
        text_inputs=batch["text"]
        text_inputs={key: value.to(DEVICE) for key,value in text_inputs.items()}

        with torch.no_grad():
            preds=model(**text_inputs)
            predict_list+=preds.cpu().detach().numpy().tolist()
    
    predictions=np.argmax(predict_list,axis=1)
    acc=(df["label"]==predictions).mean()

    
    gc.collect()
    
    return predict_list,predictions,acc

if __name__=="__main__":
    # print(os.getcwd())
    pass
