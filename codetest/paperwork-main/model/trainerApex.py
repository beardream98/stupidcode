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
from torch.optim import AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, logging
import torch.distributed as dist

from .tricks import EarlyStopping,get_parameters
from data.datasets import make_dataloader,prepare_loaders
from .head import OriginModel
from utils.train import get_score,train_set,reduce_tensor,get_score_list,gather_tensor
from utils.train import loss_process_select
from data.read_data import TrainData
import gc 
gc.enable()


def evaluate(model,dev_dataloader,DEVICE,is_dist=False):
    model.eval()
    
    total_val_loss=0
    loss_process=loss_process_select(is_dist)
    
    for batch in tqdm(dev_dataloader):
        text_inputs=batch["text"]
        label=batch["label"].to(DEVICE,non_blocking=True)
        
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        text_inputs["labels"]=label
        with torch.no_grad():
            loss=model(**text_inputs)


        total_val_loss+=loss_process(loss)
        
    return total_val_loss/len(dev_dataloader)
def train(model,data_df,fold,CONFIG,DEVICE,run=None,trainDataObj=None):
    
    train_df,val_df=data_df[data_df.fold!=fold],data_df[data_df.fold==fold]
    train_dataloader,dev_dataloader,train_sampler,_=prepare_loaders(data_df,fold,CONFIG)
    print("loader prepared works")
    start_epoch=0
    best_val_loss=100
    best_val_acc=-1
    best_val_f1=-1

    es=EarlyStopping(patience=CONFIG["early_patience"],mode="max")
    continue_path=os.path.join(CONFIG["ContinuePath"],f"LastCheckpoint{fold}.pth")
    """
    get_linear_schedule_with_warmup:学习率先从0开始warm_up到设定学习率，再逐渐减到0
    num_warmup_steps：完成预热的步数
    num_training_steps：训练批次*CONFIG["EPOCHS"] 训练的step数
    """
    
    num_training_steps=len(train_dataloader)*CONFIG["EPOCHS"]
    num_warm_steps=int(CONFIG["NUM_WARM_RATE"]*num_training_steps)
    
    model.to(DEVICE,non_blocking=True)
    optimizer=AdamW(model.parameters(),lr = CONFIG["LR"]) 
    if CONFIG["DIST_MODEL"]:
        args=CONFIG["args"]
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
                output_device=args.local_rank,find_unused_parameters=True)
        dist.barrier()
    if CONFIG["ContinueTrain"]:
        checkpoint=torch.load(continue_path)
        start_epoch=checkpoint["epoch"]+1
        best_val_loss=checkpoint["metric"]["loss"]
        best_val_acc=checkpoint["metric"]["acc"]
        best_val_f1=checkpoint["metric"]["f1"]
        trainDataObj.iter_now=checkpoint["iter_now"]
        es.set_state(checkpoint["es"]["best"],checkpoint["es"]["num_bad_epochs"])
        CONFIG["Logger"].info(f"earlystop: best: {checkpoint['es']['best']} \
            num_bad_epochs:{checkpoint['es']['num_bad_epochs']}")
        
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer=AdamW(model.parameters(),lr = CONFIG["LR"])    
        # optimizer=AdamW([{'params': model.parameters(), 'initial_lr': CONFIG["LR"]}], lr = CONFIG["LR"])    
#       optimizer=AdamW(get_parameters(model, model_init_LR=CONFIG["LR"]*1.5, multiplier=0.975, classifier_LR=CONFIG["LR"]), LR = CONFIG["LR"], EPS= CONFIG["EPS"])
        optimizer.load_state_dict(checkpoint["opt_state_dict"])

        CONFIG["Logger"].info(f"learning rate start from {optimizer.param_groups[0]['lr']}")
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warm_steps, \
                                                num_training_steps=len(train_dataloader) * CONFIG["EPOCHS"],last_epoch=start_epoch-1)
        scheduler.load_state_dict(checkpoint["scheduler"])
        CONFIG["Logger"].info(f"learning rate before scheduler.step() {optimizer.param_groups[0]['lr']}",)
        
        # 将学习率进行对齐上一个checkpoint
        scheduler.step()
        CONFIG["Logger"].info(f"use the dict from preview model at epoch {checkpoint['epoch']+1}")
        CONFIG["Logger"].info(f"learning rate start from {optimizer.param_groups[0]['lr']}",)


    else:
        # optimizer=AdamW([{'params': model.parameters(), 'initial_lr': CONFIG["LR"]}], lr = CONFIG["LR"])    
#       optimizer=AdamW(get_parameters(model, model_init_LR=CONFIG["LR"]*1.5, multiplier=0.975, classifier_LR=CONFIG["LR"]), LR = CONFIG["LR"], EPS= CONFIG["EPS"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warm_steps, \
                                                num_training_steps=len(train_dataloader) * CONFIG["EPOCHS"],last_epoch=start_epoch-1)
    
    scaler = GradScaler(enabled=CONFIG["FP16"])

    if CONFIG["run_db"]==True:
        wandb.watch(model,log_freq=100)
    
    CONFIG["Logger"].info("-"*10+"train start"+"-"*10)

    loss_process=loss_process_select(is_dist=CONFIG["DIST_MODEL"])
    for epoch in range(start_epoch,CONFIG["EPOCHS"]):
        CONFIG["Logger"].info(f"learning rate at epoch start {optimizer.param_groups[0]['lr']}",)
        sum_train_loss=0
        start = time.time()
        model.train()
        if epoch!=0 and trainDataObj!=None:
            data_df=trainDataObj.get_data()
            train_df,val_df=data_df[data_df.fold!=fold],data_df[data_df.fold==fold]
            train_dataloader,dev_dataloader,train_sampler,_=prepare_loaders(data_df,fold,CONFIG)
        if CONFIG["DIST_MODEL"]:
            train_sampler.set_epoch(epoch)

        for index,batch in enumerate(tqdm(train_dataloader,leave=False)):
            
            model.zero_grad()
            text_inputs=batch["text"]
            label=batch["label"].to(DEVICE,non_blocking=True)
        
            text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
            text_inputs["labels"]=label
            with autocast(enabled=CONFIG["FP16"]):
                loss=model(**text_inputs)

            sum_train_loss+=loss_process(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scaler.update()

#             loss.backward()
#             optimizer.step()
            scheduler.step()

        avg_train_loss=sum_train_loss/(len(train_dataloader))
        val_loss=evaluate(model,dev_dataloader,DEVICE,is_dist=CONFIG["DIST_MODEL"])
        
        _,predictions,val_acc,val_labels=predict(val_df,DEVICE,CONFIG,None,model)
        precesion,recall,f1=get_score_list(val_labels,predictions,CONFIG["average"],position=1)

        CONFIG["Logger"].info(f"EPOCH:{epoch+1}/{CONFIG['EPOCHS']} train_loss:{avg_train_loss},val_loss:{val_loss},\
            f1:{f1},val_acc:{val_acc}")
        
        time_used = time.time()-start
        CONFIG["Logger"].info(f"one_epoch_time:{time_used}")
        
        if val_acc>best_val_acc:
            best_val_acc=val_acc

        if val_loss<best_val_loss:
            best_val_loss=val_loss
        if CONFIG["args"].local_rank==0:
            if f1>best_val_f1:
                best_val_f1=f1
                model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
                torch.save(model.state_dict(),model_path)
                CONFIG["Logger"].info(f"best_model saved ,f1:{best_val_f1},recall:{recall},val acc:{val_acc}")

        if CONFIG["run_db"]==True:
            wandb.log({"Train LOSS":avg_train_loss,"Valid LOSS":val_loss})
            wandb.log({"Valid ACC":val_acc})
            wandb.log({"Valid F1":f1})
            run.summary["best_val_loss"]=best_val_loss
            run.summary["best_val_acc"]=best_val_acc
            run.summary["best_val_f1"]=best_val_f1
            
        if es.step(f1):
            break
        if CONFIG["args"].local_rank==0:
            if (epoch+1)%CONFIG["save_checkpoint_iter"]==0:
                CONFIG["Logger"].info("-"*10+f"save checkpoint at epoch: {epoch+1}"+"-"*10)
                CONFIG["Logger"].info(f"save learning rate :{optimizer.param_groups[0]['lr']}")
                CONFIG["Logger"].info(f"save es : {es.get_state()}")

                metirc_dict={"loss":best_val_loss,"acc":best_val_acc,"f1":best_val_f1}
                checkpoint={
                    "metric":metirc_dict,
                    "model_state_dict":model.state_dict(),
                    "epoch":epoch,
                    "opt_state_dict":optimizer.state_dict(),
                    "es":es.get_state(),
                    "iter_now":trainDataObj.iter_now,
                    "scheduler":scheduler.state_dict()
                }

                torch.save(checkpoint,continue_path)

    CONFIG["Logger"].info("-"*10+"train finished"+"-"*10)
    
    CONFIG["Logger"].info(f"best_val_acc:{best_val_acc},best_val_loss:{best_val_loss},best_val_f1:{best_val_f1}")
    return best_val_loss

def predict(df,DEVICE,CONFIG,model_path=None,model=None):
    if model==None:
        CONFIG["Logger"].info(f"\nUsing{model_path}")
        model=OriginModel(CONFIG)
        model.to(DEVICE,non_blocking=True)
        if CONFIG["DIST_MODEL"]:
            args=CONFIG["args"]
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
                output_device=args.local_rank,find_unused_parameters=True)
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    #set is_test=True to get label value
    data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],\
        CONFIG["MAX_LENGTH"],is_test=False,is_dist=CONFIG["DIST_MODEL"])

    predict_list,label_list=[],[]
    for batch in tqdm(data_loader):
        text_inputs=batch["text"]
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        labels=batch["label"].to(DEVICE,non_blocking=True)

        with torch.no_grad():
            preds=model(**text_inputs)
            
        predict_list.append(preds)
        label_list.append(labels)
    
        # predict_list+=preds.cpu().detach().numpy().tolist()
        # label_list+=labels.cpu().detach().numpy().tolist()
    predict_tensor,label_tensor=torch.cat(predict_list,dim=0),torch.cat(label_list,dim=0)
    
    if CONFIG["DIST_MODEL"]:
        dist.barrier()
        predict_tensor=gather_tensor(predict_tensor,DEVICE=DEVICE,world_size=CONFIG["world_size"],dtype=torch.float32)
        label_tensor=gather_tensor(label_tensor,DEVICE=DEVICE,world_size=CONFIG["world_size"],dtype=torch.long)
        predict_tensor,label_tensor=torch.cat(predict_tensor,dim=0),torch.cat(label_tensor,dim=0)
    

    predict_list=predict_tensor.cpu().detach().numpy().tolist()
    label_list=label_tensor.cpu().detach().numpy().tolist()

    predictions=np.argmax(predict_list,axis=1)
    ## may cause error 
    acc=(np.asarray(label_list)==predictions).mean()

    
    gc.collect()
    
    return predict_list,predictions,acc,label_list

if __name__=="__main__":
    # print(os.getcwd())
    pass
