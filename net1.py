from torch import nn
import torch

#构建网络
class Net(nn.Module):
    #构建前向网络
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, 1),
            nn.ReLU(),
            nn.Conv2d(20, 40, 3, 1),
            nn.ReLU(),
            nn.Conv2d(40, 80, 3, 1),
            nn.ReLU(),
        )

        self.layers2 = nn.Linear(8000*4, 10)

    # 前向计算
    def forward(self,x):
        conv_out = self.layers(x)
        # print(conv_out.shape)
        conv_out = conv_out.reshape(-1,8000*4)
        # print(type(conv_out))
        conv_out2 = self.layers2(conv_out)
        return conv_out2
        pass

if __name__ == '__main__':
    a = torch.randn(1,1,28,28)
    net = Net()
    b = net(a)
    print(b.shape)
    pass
