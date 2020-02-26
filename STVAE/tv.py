import torch
import torchvision
from torchvision import transforms as TR
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

c10=torchvision.datasets.CIFAR10("CT", train=True,
        transform=TR.ToTensor(), target_transform=None, download=True)


data_loader = torch.utils.data.DataLoader(c10,batch_size=50000)

OUT=[]
tt=TR.Compose([TR.ToPILImage(),TR.RandomAffine(20),TR.ToTensor()])


for d,l in data_loader:
        inp, lab = d, l

t1=time.time()
out=torch.zeros_like(inp)
for j, inpa in enumerate(inp):
        out[j] = tt(inpa)


print('{0:5.3f}s'.format(time.time() - t1))




print("hello")

