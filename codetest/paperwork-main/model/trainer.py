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
from torch.optim import AdamW
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, logging
import torch.distributed as dist
from apex import amp


from .tricks import EarlyStopping,get_parameters
from data.datasets import make_dataloader,prepare_loaders
from .head import OriginModel,LR,get_model
from utils.train import get_score,train_set,reduce_tensor,get_score_list,gather_object,gather_tensor
from utils.train import loss_process_select,remove_module
from data.read_data import TrainData
import gc 

from memory_profiler import profile

gc.enable()

# @profile(precision=4,stream=open('memory_profiler_evaluate.log','w+'))
def evaluate(model,val_dataloader,DEVICE,is_dist=False):
    model.eval()
    
    total_val_loss=0
    loss_process=loss_process_select(is_dist)
    
    for batch in tqdm(val_dataloader):
        text_inputs=batch["text"]
        label=batch["label"].to(DEVICE,non_blocking=True)
        
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        text_inputs["labels"]=label
        with torch.no_grad():
            loss=model(**text_inputs)

        total_val_loss+=loss_process(loss)
        
    return total_val_loss/len(val_dataloader)
def predict_evaluate(val_df,val_dataloader,model,DEVICE,CONFIG):

    model.eval()
    total_val_loss=0
    loss_process=loss_process_select(CONFIG["DIST_MODEL"])
    criterion = nn.CrossEntropyLoss()

    predict_list,index_list=[],[]
    for batch in tqdm(val_dataloader):
        text_inputs=batch["text"]
        label=batch["label"].to(DEVICE,non_blocking=True)
        index=batch["index"]
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        with torch.no_grad():
            index=batch["index"]
        with torch.no_grad():
            preds=model(**text_inputs)
            loss=criterion(preds, label)
            preds=F.softmax(preds,dim=1)

        total_val_loss+=loss_process(loss)
        
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

    val_df["predict"]=-1
    val_df.loc[index_array,"prob0"]=predict_array[:,0]
    val_df.loc[index_array,"prob1"]=predict_array[:,1]
    val_df.loc[index_array,"predict"]=predictions

    if CONFIG["DIST_MODEL"]:
        dist.barrier()
    ## may cause error 
    acc=(val_df["label"]==val_df["predict"]).mean()

    del predictions,index_array,predict_array,predict_list,index_list
    return acc,total_val_loss/len(val_dataloader)



# @profile(precision=4,stream=open('memory_profiler_predict.log','w+'))
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
            model.load_state_dict(torch.load(model_path))
        else:
            ### may cause error 
            
            state_dict=remove_module(torch.load(model_path))
            model.load_state_dict(state_dict)
        
    model.eval()
    #set is_test=True to get label value
    data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],CONFIG["MAX_LENGTH"],\
        is_test = False,is_dist = CONFIG["DIST_MODEL"],load_tokenizer = CONFIG["SET_EMBEDDING"] ,tokenizer_dir = CONFIG["INPUT_DIR"])
        
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

    del predictions,index_array,predict_array,predict_list,index_list
    return df,acc

# @profile(precision=4,stream=open('memory_profiler_trainer.log','w+'))
def trainOneEpoch(model,CONFIG,train_dataloader,val_dataloader,val_df,DEVICE,loss_process,optimizer,scheduler):

    sum_train_loss=0
    for index,batch in enumerate(tqdm(train_dataloader,leave=False)):
        
        model.zero_grad()
        text_inputs=batch["text"]
        label=batch["label"].to(DEVICE,non_blocking=True)
    
        text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
        text_inputs["labels"]=label
        loss=model(**text_inputs)
        sum_train_loss+=loss_process(loss)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss=sum_train_loss/(len(train_dataloader))

    val_acc,val_loss=predict_evaluate(val_df,val_dataloader,model,DEVICE,CONFIG)

    # val_loss=evaluate(model,val_dataloader,DEVICE,is_dist=CONFIG["DIST_MODEL"])
    # val_df,val_acc=predict(val_df,DEVICE,CONFIG,None,model)

    val_labels,predictions=val_df["label"].values.tolist(),val_df["predict"].values.tolist()
    # predict_list,predictions,val_acc,val_labels=predict(val_df,DEVICE,CONFIG,None,model)

    precesion,recall,f1=get_score_list(val_labels,predictions,CONFIG["average"],position=1)
    return avg_train_loss,val_loss,val_acc,precesion,recall,f1

# @profile(precision=4,stream=open('memory_profiler_train.log','w+'))
def train(model,data_df,fold,CONFIG,DEVICE,run=None,trainDataObj=None):
    
    train_dataloader,val_dataloader,train_sampler,_,val_df=prepare_loaders(data_df,fold,CONFIG)

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
    # num_training_steps=CONFIG["EPOCHS"]//CONFIG["early_patience"]

    model.to(DEVICE,non_blocking=True)
    optimizer=AdamW(model.parameters(),lr = CONFIG["LR"])    
    model, optimizer = amp.initialize(model, optimizer, opt_level=CONFIG["FP16"])
    
    if CONFIG["DIST_MODEL"]:
        args=CONFIG["args"]
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
                output_device=args.local_rank,find_unused_parameters=True)
        dist.barrier()

    
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warm_steps, \
                                            num_training_steps=num_training_steps,last_epoch=start_epoch-1)

    # scaler = GradScaler(enabled=CONFIG["FP16"])

    CONFIG["Logger"].info("-"*10+"train start"+"-"*10)

    loss_process=loss_process_select(is_dist=CONFIG["DIST_MODEL"])
    
    for epoch in range(start_epoch,CONFIG["EPOCHS"]):
        CONFIG["Logger"].info(f"epoch {epoch} learning rate start  :{optimizer.param_groups[0]['lr']}")
        start = time.time()
        model.train()
        # if epoch!=0 and trainDataObj!=None and CONFIG["different_negative"]:
        #     data_df=trainDataObj.get_data()
        #     train_dataloader,val_dataloader,train_sampler,_,val_df=prepare_loaders(data_df,fold,CONFIG)
        if CONFIG["DIST_MODEL"]:
            dist.barrier()
            train_sampler.set_epoch(epoch)

        avg_train_loss,val_loss,val_acc,precesion,recall,f1=\
            trainOneEpoch(model,CONFIG,train_dataloader,val_dataloader,val_df,DEVICE,loss_process,optimizer,scheduler)

        CONFIG["Logger"].info(f"EPOCH:{epoch+1}/{CONFIG['EPOCHS']} train_loss:{avg_train_loss},val_loss:{val_loss}")
        CONFIG["Logger"].info(f"f1:{f1},recall:{recall},precesion:{precesion},val_acc:{val_acc}")
        
        time_used = time.time()-start
        CONFIG["Logger"].info(f"one_epoch_time:{float(time_used)/60} min")
        
        if val_acc>best_val_acc:
            best_val_acc=val_acc
        if val_loss<best_val_loss:
            best_val_loss=val_loss
        if CONFIG["args"].local_rank==0:
            if f1>best_val_f1:
                best_val_f1=f1
                model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
                torch.save(model.state_dict(),model_path)
                CONFIG["Logger"].info(f"best_model saved ,f1:{best_val_f1},recall:{recall},precesion:{precesion},val acc:{val_acc}")
        if run!=None:
            wandb.log({"Train LOSS":avg_train_loss,"Valid LOSS":val_loss})
            wandb.log({"Valid ACC":val_acc})
            wandb.log({"Valid F1":f1})
            run.summary["best_val_loss"]=best_val_loss
            run.summary["best_val_acc"]=best_val_acc
            run.summary["best_val_f1"]=best_val_f1
        if es.step(f1):
            # scheduler.step()
            # CONFIG["Logger"].info(f"generate new lr :{optimizer.param_groups[0]['lr']}")
            # es.num_bad_epochs=0
            break

    CONFIG["Logger"].info("-"*10+"train finished"+"-"*10)
    
    CONFIG["Logger"].info(f"best_val_acc:{best_val_acc},best_val_loss:{best_val_loss},best_val_f1:{best_val_f1}")



# def predict(df,DEVICE,CONFIG,model_path=None,model=None):
#     if model==None:
#         CONFIG["Logger"].info(f"\nUsing{model_path}")
#         model=OriginModel(CONFIG)
#         model.to(DEVICE,non_blocking=True)
#         if CONFIG["DIST_MODEL"]:
#             args=CONFIG["args"]
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],\
#                 output_device=args.local_rank,find_unused_parameters=True)
#             dist.barrier()
#         ### may cause error 
#         model.load_state_dict(torch.load(model_path))

#     model.eval()
#     #set is_test=True to get label value
#     data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],\
#         CONFIG["MAX_LENGTH"],is_test=False,is_dist=CONFIG["DIST_MODEL"])

#     predict_list,label_list=[],[]
#     for batch in tqdm(data_loader):
#         text_inputs=batch["text"]
#         text_inputs={key: value.to(DEVICE,non_blocking=True) for key,value in text_inputs.items()}
#         labels=batch["label"].to(DEVICE,non_blocking=True)

#         with torch.no_grad():
#             preds=model(**text_inputs)

#         predict_list.append(preds)
#         label_list.append(labels)

#     with torch.no_grad():
#         predict_tensor,label_tensor=torch.cat(predict_list,dim=0),torch.cat(label_list,dim=0)

#         if CONFIG["DIST_MODEL"]:
#             label_tensor=label_tensor.float()
#             label_tensor=torch.unsqueeze(label_tensor,1)
#             predict_label_tensor=torch.cat([label_tensor,predict_tensor],axis=1)

#             dist.barrier()
#             predict_label_tensor=gather_tensor(predict_label_tensor,DEVICE=DEVICE,world_size=CONFIG["world_size"],dtype=torch.float32)
#             # label_tensor=gather_tensor(label_tensor,DEVICE=DEVICE,world_size=CONFIG["world_size"],dtype=torch.long)
#             # predict_tensor,label_tensor=torch.cat(predict_tensor,dim=0),torch.cat(label_tensor,dim=0)

#             predict_label_tensor=torch.cat(predict_label_tensor,dim=0)
#             # torch.split is different with np.split 
#             label_tensor,predict_tensor=torch.split(predict_label_tensor,split_size_or_sections=[1,2],dim=1)
#             label_tensor=label_tensor.long()
#             label_tensor=torch.squeeze(label_tensor)

#         # predict_list+=preds.cpu().detach().numpy().tolist()
#         # label_list+=labels.cpu().detach().numpy().tolist()

#     predict_list=predict_tensor.cpu().detach().numpy().tolist()
#     label_list=label_tensor.cpu().detach().numpy().tolist()

#     predictions=np.argmax(predict_list,axis=1)
#     ## may cause error 
#     acc=(np.asarray(label_list)==predictions).mean()


#     gc.collect()

#     return predict_list,predictions,acc,label_list

if __name__=="__main__":
    # print(os.getcwd())
    pass
