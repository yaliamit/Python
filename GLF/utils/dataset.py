"""dataset.py"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
    

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    #image_size = args.image_size

    if name.lower() == 'cifar':
        root = os.path.join(dset_dir, 'cifar')
        if args.loss_type == "MSE":
            training_data = datasets.CIFAR10(root=root, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                            ]))
        else:
            training_data = datasets.CIFAR10(root=root, train=True, download=True,
                                        transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]))
    
    elif name.lower() == 'mnist':
        root = os.path.join(dset_dir, 'mnist')
        if not args.loss_type == 'Perceptual':
            training_data = datasets.MNIST(root=root, download=True, transform=transforms.ToTensor())
        else:
            training_data = datasets.MNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                      ]))
    elif name.lower() == 'fashion-mnist':
        root = os.path.join(dset_dir, 'fmnist')
        if not args.loss_type == 'Perceptual':
            training_data = datasets.FashionMNIST(root=root, download=True, transform=transforms.ToTensor())
        else:
            training_data = datasets.FashionMNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                      ]))    
        
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        if args.loss_type == 'MSE':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        else:
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
        training_data = dset(**train_kwargs)
    else:
        raise NotImplementedError

        
    train_loader = DataLoader(training_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)
    return train_loader

