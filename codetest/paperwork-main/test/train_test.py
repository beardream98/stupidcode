import yaml
import sys
sys.path.insert(0,"..")
import data
from data.read_data import select_data,TrainData

from data.data_split import kfsplit
from data.datasets import prepare_loaders
from utils.wandb_init import get_run,wandb_utils
from utils.train import get_score,train_set
from utils.trivial import get_logger

import torch
from transformers import logging
from model.head import OriginModel
from model.trainer import train,predict

import wandb

import numpy as np
import pandas as pd 
import os
import torch

from sklearn.metrics import precision_score,recall_score,f1_score
import gc 
gc.enable()
from utils.trivial import set_seed



yaml_path="../config/train.yaml"
CONFIG,Logger,DEVICE=train_set(yaml_path,experimentName=None,upload=True)

CONFIG["run_db"]=False
CONFIG["DEBUG_MODEL"]=True
CONFIG["num_hidden_layers"]=3
CONFIG["DATA_NUM"]=3000
CONFIG["EPOCHS"]=3

data_df,test_df=select_data(CONFIG)
# data_df=kfsplit(data_df,CONFIG)
trainDataObj=TrainData(data_df,CONFIG)
train_data=trainDataObj.get_data()
if not CONFIG["different_negative"]:
    trainDataObj=None

fold=0
CONFIG["Logger"].info(f"----------fold {fold}----------")
run=get_run(CONFIG,"Train",fold)
model=OriginModel(CONFIG)
model.to(DEVICE)

val_loss,best_model_dict=train(model,train_data,fold,CONFIG,DEVICE,run=run,trainDataObj=trainDataObj)

# model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
# torch.save(best_model_dict,model_path)
_,_,acc_train_lastepoch=predict(data_df,DEVICE,CONFIG,model_path=None,model=model)

CONFIG["Logger"].info(f"train{fold} acc in last epoch:{acc_train_lastepoch}")
del model
torch.cuda.empty_cache()
if run!=None:
    run.finish()
