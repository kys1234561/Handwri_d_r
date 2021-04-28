import torch
from PIL import Image
from torchvision import transforms
import net1

img = Image.open(r'E:\practice_project\手写数字识别\MNIST_IMG\TEST\1\100.jpg')
# img.show('img', img)
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)
net = net1.Net()
net.load_state_dict(torch.load('param/10.t'))
out = net(img)
y = torch.argmax(out)
print(out)
print(y)