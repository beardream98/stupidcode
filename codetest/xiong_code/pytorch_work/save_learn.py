import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer


net=torch.nn.Sequential(
    torch.nn.Linear(2,10),torch.nn.ReLU(),torch.nn.Linear(10,2)
)
optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
loss_fun=torch.nn.CrossEntropyLoss()
plt.ion()   # something about plotting

for t in range(100):
    out=net(x)
    loss=loss_fun(out,y)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]  #第一个是value 我们不需要 我们要的只是value 所在的索引 代表着标签
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

plt.ioff()


def save():
    torch.save(net,'net.pkl')
    torch.save(net.state_dict,'net_params.pkl')
save()
def restore_net():
    net2=torch.load('net.pkl')
def restore_params():
    net3=torch.nn.Sequential(torch.nn.Linear(2,10),torch.nn.ReLU(),torch.nn.Linear(10,2))
    net3.load_state_dict('net_params.pkl')
    