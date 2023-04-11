from sklearn.metrics import precision_score,recall_score,f1_score
import yaml
from transformers import logging
from utils.wandb_init import get_run,wandb_utils
from utils.trivial import get_logger
import torch
import argparse
import torch.distributed as dist
from collections import OrderedDict
from torch import distributed
import os

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt
def gather_tensor(tensor: torch.Tensor,world_size:int,DEVICE,dtype=torch.float32) ->torch.Tensor:
    tensor_list=[torch.zeros(tensor.size(),dtype=dtype).to(DEVICE) for _ in range(world_size)]
    dist.all_gather(tensor_list,tensor)
    
    return tensor_list

@torch.no_grad()
def gather_object(obj,world_size:int,group=None):
    dist.barrier()
    gather_obj=[None for _ in range(world_size)]
    dist.all_gather_object(gather_obj,obj)
    dist.barrier()
    return gather_obj

def loss_process_select(is_dist):
    def get_loss_item(loss):
        return loss.item()

    def get_loss_reduce(loss):
        dist.barrier()
        reduce_loss=reduce_tensor(loss.data)
        return reduce_loss.item()
    if is_dist:
        return get_loss_reduce
    else:
        return get_loss_item


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v
    return new_state_dict


def get_score(df,predictions,average="macro",position=1):

    if average!=None:

        precesion=precision_score(df["label"], predictions, average=average)
        recall=recall_score(df["label"], predictions, average=average)
        f1=f1_score(df["label"], predictions, average=average)
    else:
        precesion=precision_score(df["label"], predictions, average=None)[position]
        recall=recall_score(df["label"], predictions, average=None)[position]
        f1=f1_score(df["label"], predictions, average=None)[position]
    return precesion,recall,f1
def get_score_list(labels,predictions,average="macro",position=1):

    if average!=None:

        precesion=precision_score(labels, predictions, average=average)
        recall=recall_score(labels, predictions, average=average)
        f1=f1_score(labels, predictions, average=average)
    else:
        precesion=precision_score(labels, predictions, average=None)[position]
        recall=recall_score(labels, predictions, average=None)[position]
        f1=f1_score(labels, predictions, average=None)[position]
    return precesion,recall,f1

def read_yaml(yaml_path):
    
    with open(yaml_path,"r",encoding="UTF-8") as file_object:
        file_data=file_object.read()
    
    yaml_dict=yaml.load(file_data,Loader=yaml.FullLoader)

    return yaml_dict
def parse(is_notebook=False):
    parser=argparse.ArgumentParser()
    parser.add_argument("--local_rank",default=0,type=int)
    #may cause error 
    if is_notebook:
        args=parser.parse_args(args=[])
    else:
        args=parser.parse_args()
    return args
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def train_set(yaml_path,experimentName=None,upload=True,filename=None,is_notebook=False):

    logging.set_verbosity_error()

    CONFIG=read_yaml(yaml_path)
    if CONFIG["DEBUG_MODEL"]:
        filename="./test/logs"
        # CONFIG["run_db"]=False
    local_rank=0
    args=parse(is_notebook)
    if 'WORLD_SIZE' in os.environ:
        CONFIG["DIST_MODEL"]=True
    else:
        CONFIG["DIST_MODEL"]=False
    CONFIG["args"]=args

    if CONFIG["DIST_MODEL"]:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        dist.barrier()
        world_size=dist.get_world_size()
        CONFIG["LR"]=CONFIG["LR"]*(int(world_size)//2)
        CONFIG["world_size"]=world_size
        print(CONFIG["args"].local_rank)
        setup_for_distributed(args.local_rank==0)
        local_rank=args.local_rank
    else:
        CONFIG["world_size"]=1
    
    Logger=get_logger(filename,rank=local_rank)
    CONFIG["Logger"]=Logger
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda", args.local_rank)
    else:
        DEVICE = torch.device('cpu')

    if CONFIG["args"].local_rank==0:
        wandb_utils(experimentName=None,upload=upload)

    CONFIG["DEVICE"] = DEVICE
    return CONFIG,Logger,DEVICE
if __name__=='__main__':
    pass