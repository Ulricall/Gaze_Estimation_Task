import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file,encoding='utf-8',delimiter=' ')
        #print(self.img_labels)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        _path=self.img_labels.iloc[idx,0]
        _path=_path.replace("\\","/")
        img_path=os.path.join(self.img_dir,_path)
        image=read_image(img_path)
        #print(image)
        label=torch.tensor(list(map(float,self.img_labels.iloc[idx,5].split(','))))
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image,label
