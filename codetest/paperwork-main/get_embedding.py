import yaml
import data
from data.read_data import select_data,TrainData,read_data_txt,sample_negative
from data.data_split import kfsplit
from data.datasets import prepare_loaders
from utils.wandb_init import get_run,wandb_utils
from utils.train import get_score,train_set
from utils.trivial import get_logger
from model.head import OriginModel,get_model
from tqdm import tqdm
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
from data.datasets import make_dataloader
def get_embedding(df,DEVICE,CONFIG,embedding_path,model_path=None,embedding_name="DataEmbeddings"):
    
    CONFIG["Logger"].info(f"\nUsing{model_path}")
    CONFIG["OUTPUT_EMBEDDING"] = True
    model=get_model(CONFIG["model_name"])(CONFIG)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
        
    model.eval()

    Query=df["Query"].values.tolist()
    index=0
    embeddings=[]
    with open(os.path.join(embedding_path,embedding_name),"w") as file_obj:
        print("clear file")
    
    data_loader,_=make_dataloader(df,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],CONFIG["MAX_LENGTH"],is_test=True)
    
    for k,batch in enumerate(tqdm(data_loader)):
        text_inputs=batch["text"]
        text_inputs={key: value.to(DEVICE) for key,value in text_inputs.items()}
        with torch.no_grad():
            embedding=model(**text_inputs)
            embeddings+=embedding.cpu().detach().numpy().tolist()
    
        # embedding_array=np.asarray(embeddings,dtype=str)
        if k%50==0 or k==len(data_loader)-1:
            with open(os.path.join(embedding_path,embedding_name),"a") as file_obj:
                
                for i in range(len(embeddings)):
                    s=Query[index]
                    embedding="|".join(list(map(str, embeddings[i])))
            #         embedding="|".join(embedding_array[i])
                    index+=1
                    s=s+"\t"+embedding+"\n"
                    file_obj.writelines(s)
            embeddings=[]
        
    del model
    gc.collect()
    return embeddings

def embedding_floop(CONFIG,test_df,DEVICE,embedding_path,save_path=None,embedding_name="DataEmbeddings"):
    for fold in range(CONFIG["NUM_FOLDS"]):
        if CONFIG["cross_val"]==False and fold>=1:
            break
        if save_path==None:
            model_path=os.path.join(CONFIG["SAVE_PATH"],f"bestmodel{fold}.pth")
        else:
            model_path=os.path.join(save_path,f"bestmodel{fold}.pth")

        embeddings=get_embedding(test_df,DEVICE,CONFIG,embedding_path,model_path,embedding_name=embedding_name)
        CONFIG["Logger"].info("Embedding finished")
    torch.cuda.empty_cache()
    gc.collect()
    return embeddings
if __name__=='__main__':


    yaml_path="./config/train.yaml"
    embedding_path="./embeddings"
    CONFIG,Logger,DEVICE=train_set(yaml_path,experimentName=None,upload=True)

    input_df=read_data_txt(CONFIG["INPUT_DIR"],CONFIG["SCHEMA_DIR"],CONFIG["DEBUG_MODEL"],CONFIG['DATA_NUM'])
    input_df["label"]=input_df["label"].replace(["0","1"],[0,1])
    input_df=sample_negative(input_df)
    CONFIG["Logger"].info("-"*10+"input_df label's value_counts"+"-"*10)
    CONFIG["Logger"].info(input_df["label"].value_counts())
    # data_df=kfsplit(data_df,CONFIG)
    set_seed(CONFIG['SEED'])

    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    embedding_floop(CONFIG,input_df,DEVICE,embedding_path)    


    
    # input_df.to_csv(os.path.join(embedding_path,"DataEmbeddings"))

