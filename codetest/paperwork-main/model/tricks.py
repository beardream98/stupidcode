import numpy as np


def get_parameters(model,model_init_LR,multiplier, classifier_LR):
    #权重分层，越靠近下游学习率越高
    parameters=[]
    LR=model_init_LR
    # 迭代器包含 层名字和参数 parameters()函数只包含参数
    #定义的层字典，参数的key必须叫params，否则在optimizer 父类中冲突
    for layer in range(model.config.to_dict()["num_hidden_layers"]-1,-1,-1):
        layer_parameters={
            "params":[p for n,p in model.named_parameters() if f"encoder.layer.{layer}." in n],
            "LR":model_init_LR
        }
        LR*=multiplier
        parameters.append(layer_parameters)
    
    
    classify_parameters={
        #自己定义了什么分类层在此更改名字
        "params":[p for n,p in model.named_parameters() if "linear"  in n],
        "LR":classifier_LR
    }
        
    parameters.append(classify_parameters)
    return parameters

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
    def set_state(self,best,num_bad_epochs):
        self.best=best
        self.num_bad_epochs=num_bad_epochs
    def get_state(self):
        return {"best":self.best,"num_bad_epochs":self.num_bad_epochs}
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

if __name__=='__main__':
    pass