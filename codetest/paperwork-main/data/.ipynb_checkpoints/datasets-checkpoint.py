import numpy as np
import pandas as pd 
import os
import random
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, SequentialSampler, RandomSampler, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, logging


import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data.distributed import DistributedSampler
from utils.trivial import timer

class DatasetRetriever(Dataset):
    def __init__(self,data,tokenizer,max_len,is_test=False):
        self.index=data.index.values.tolist()
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.text=data["Query"].values.tolist()
        self.is_test=is_test
        if self.is_test==False:
            self.label=data["label"].values.tolist()
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        text=self.text[item]
        index=self.index[item]
        features=self.convert_examples_to_features(text)
        features={"input_ids":features["input_ids"],"attention_mask":features["attention_mask"]}
        if self.is_test==False:
            label=self.label[item]
            return {"text":{key:torch.tensor(value,dtype=torch.long) for key,value in features.items()},
                    "label":torch.tensor(label,dtype=torch.long),"index":torch.tensor(index,dtype=torch.long)}
        else:
            return {"text":{key:torch.tensor(value,dtype=torch.long) for key,value in features.items()},"index":torch.tensor(index,dtype=torch.long)}
        
    def convert_examples_to_features(self, example):
        encoded = self.tokenizer.encode_plus(
            example,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=False,
            )
        return encoded

# from memory_profiler import profile
# @profile(precision=4,stream=open('memory_profiler_trainer.log','w+'))
def make_dataloader(data,batch_size,model_dir,max_len,is_test=False,is_dist=False):
    tokenizer=AutoTokenizer.from_pretrained(model_dir)
    dataset=DatasetRetriever(data,tokenizer,max_len,is_test=is_test)
    if is_dist:
        sampler=DistributedSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)
    # when the datanum in last batch < len_gpus , get an error
    dataloader=DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=8,
                        pin_memory=False
                        )
    return dataloader,sampler

def prepare_loaders(df,fold,CONFIG):
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)
    train_loader,train_sampler=make_dataloader(df_train,CONFIG["TRAIN_BATCH_SIZE"],CONFIG["model_name"],CONFIG["MAX_LENGTH"],is_dist=CONFIG["DIST_MODEL"])
    
    valid_loader,valid_sampler=make_dataloader(df_valid,CONFIG["DEV_BATCH_SIZE"],CONFIG["model_name"],CONFIG["MAX_LENGTH"],is_dist=CONFIG["DIST_MODEL"])
    
    return train_loader,valid_loader,train_sampler,valid_sampler


if __name__=='__main__':
    pass
