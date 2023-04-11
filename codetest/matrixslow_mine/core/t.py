import sys
sys.path.append("..")
from core.module import MSELoss
from core.module import Linear
from core.opt import SGD
from core.node import Tensor


model=Linear(1,1)

optimizer=SGD(model.parameters(),lr=0.0001)
loss=MSELoss("mean")

# 面积
areas = [64.4, 68, 74.1, 74., 76.9, 78.1, 78.6]
# 挂牌售价
prices = [6.1, 6.25, 7.8, 6.66, 7.82, 7.14, 8.02]

X = Tensor(areas).reshape((-1, 1))
y = Tensor(prices).reshape((-1, 1))


epochs = 1000
losses = []
for epoch in range(epochs):
    output=model(X)
    l=loss(output,y)

    optimizer.zero_grad()

    l.backward()
    optimizer.step()

    losses.append(l.data)
    if (epoch+1)%20==0:

        print(f'epoch {epoch + 1}, loss {float(l.data):f}')