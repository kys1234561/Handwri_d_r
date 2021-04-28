from torch.utils.data import DataLoader
import torch
import net1
from torchvision import datasets#导入含有数据集的包
from torchvision import transforms
from torch import optim
from torch.nn.functional import one_hot
from torch import nn
from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda:0'

class Train():
    def __init__(self, root):
        #创建可视化对象
        self.summarywriter = SummaryWriter('logs')
        #加载数据
        self.train_data = datasets.MNIST(root, True, transforms.ToTensor())
        self.train_dataloader = DataLoader(self.train_data, 100, True)
        self.test_data = datasets.MNIST(root, False, transforms.ToTensor())
        self.test_dataloader = DataLoader(self.test_data, 100, True)

        #构建网络
        self.net = net1.Net()
        #将网络放到cuda
        self.net.to(DEVICE)
        #构建优化器
        self.opt = optim.Adam(self.net.parameters())
        #定义损失函数
        self.mseloss = nn.MSELoss()
        # print(self.mseloss)

    def __call__(self):
        k = 0
        for epoch in range(1000):
            sum_loss = 0
            for i, (img, target) in enumerate(self.train_dataloader):
                img = img.to(DEVICE)
                target = target.to(DEVICE)
                out = self.net(img)
                tar = one_hot(target, 10).float()
                # loss = torch.mean((out-tar)**2)
                self.loss = self.mseloss(out,tar)

                #三部曲
                self.opt.zero_grad()#
                self.loss.backward()#反向传播
                self.opt.step()#梯度更新

                sum_loss += self.loss.item()
                self.summarywriter.add_scalar('loss', self.loss.item(), k)
                k += 1
            avg_loss = sum_loss / len(self.train_dataloader)
            print('平均损失是：',avg_loss,'轮次是：',epoch)
            #进入测试
            # self.net.eval()
            torch.save(self.net.state_dict(), f"param/{epoch}.t")  # 保留模型参数
            count = 0
            for i, (img, target) in enumerate(self.test_dataloader):
                self.net.eval()
                img = img.to(DEVICE)
                target = target.to(DEVICE)
                tar = one_hot(target, 10)
                out = self.net(img)

                out1 = torch.argmax(out, dim = 1)#torch.argmax获取最大值的索引。
                targ = torch.argmax(tar, dim = 1)
                count1 = torch.sum(torch.eq(out1,targ))
                count += count1.item()
            avg_count = count / len(self.test_dataloader)
            print('测试集的准确率：', avg_count)



if __name__ == '__main__':
    train = Train('data')
    train()
    pass
