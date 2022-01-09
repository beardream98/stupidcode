import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

# hyper parameters
LR=0.01
BACTH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(x.size()))

# plt.scatter(x.numpy(),y.numpy())
# plt.show()
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BACTH_SIZE,shuffle=True,num_workers=2 )

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
    def forward(self,x):
        x=torch.relu(self.hidden(x))
        x=self.predict(x)
        return x
if __name__ == '__main__':
    net_SGD=Net()
    net_Momentum=Net()
    net_RMSprop=Net()
    net_Adam=Net()
    nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

    opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
    opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]
    
    loss_func=torch.nn.MSELoss()
    losser_his=[[],[],[],[]] #record loss

    for epoch in range(EPOCH):
        print(epoch)
        for step,(batch_x,batch_y) in enumerate (loader):
            for net,opt,l_his in zip(nets,optimizers,losser_his): 
                output=net(batch_x)
                loss=loss_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
    labels=['SGD','Momentum','RMSprop','Adam']
    for i,l_his in enumerate(losser_his):
        plt.plot(l_his,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0,0.2))
    plt.show()
    # optimizer=torch.optim.SGD()
    # torch.optim