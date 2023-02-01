
import os
import sys
import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import pickle






class AkMinitImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir_dataset, str_trainortestorinducing):
        #grab args ===
        self.rootdir_dataset = rootdir_dataset
        self.str_trainortestorinducing = str_trainortestorinducing
        #make internals ==
        assert(isinstance(str_trainortestorinducing, str))
        assert(self.str_trainortestorinducing in [
            "train", "test", "inducing"
        ])
        fname_train = "MiniImagenet/miniImageNet_category_split_train_phase_train.pickle"
        fname_test = "MiniImagenet/miniImageNet_category_split_train_phase_test.pickle"
        #"MiniImagenet/miniImageNet_category_split_test.pickle"
        
        fname_pkl = fname_test if(str_trainortestorinducing == "test") else fname_train
        with open(os.path.join(self.rootdir_dataset, fname_pkl), 'rb') as f:
            content_pkl = pickle.load(f, encoding='latin1')
            self.X = content_pkl['data'] #[N x 84 x 84 x 3]
            self.Y = content_pkl['labels']
            
        #make the transforms ===
        tfm_colornormalization = torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
        if(self.str_trainortestorinducing == "train"):
            self.tfm = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                tfm_colornormalization
            ])
        elif(self.str_trainortestorinducing == "inducing"):
            self.tfm = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                tfm_colornormalization
            ])
        elif(self.str_trainortestorinducing == "test"):
            self.tfm = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                tfm_colornormalization
            ])
        else:
            raise Exception("Unknown str_trainortestorinducing: {}".format(
                self.str_trainortestorinducing
            ))
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, n):
        xn = self.X[n,:,:,:] #[84x84x3]
        yn = self.Y[n]
        return self.tfm(xn), yn, n
    
            
