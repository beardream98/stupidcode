import numpy as np
import torch 
import os
from logging import getLogger, INFO,WARNING, StreamHandler, FileHandler, Formatter

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from tkinter import N

def get_time():
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    return str(beijing_now)[:18]

def get_logger(filename=None,rank=0):
    if filename==None:
        filename="./logs"
    if not os.path.exists(filename):
        os.makedirs(filename)
    time=get_time()
    filename=os.path.join(filename,time)

    logger = getLogger(__name__)
    if rank in [-1,0]:
        logger.setLevel(INFO)
    else:
        logger.setLevel(WARNING)
    
    # logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)