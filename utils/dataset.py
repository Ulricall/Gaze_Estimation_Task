import os
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.io import read_image

class MPII(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, encoding = 'utf-8', delimiter = ' ')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        images = []

        face_path = self.img_labels.iloc[index, 0]
        left_path = self.img_labels.iloc[index, 1]
        right_path = self.img_labels.iloc[index, 2]
            
        face_path = face_path.replace("\\", "/")
        left_path = left_path.replace("\\", "/")
        right_path = right_path.replace("\\", "/")
            
        img_path = [os.path.join(self.img_dir, face_path), 
                    os.path.join(self.img_dir, left_path),
                    os.path.join(self.img_dir, right_path)]

        for img in img_path:
            i = read_image(img)
            if self.transform:
                i = self.transform(i)
            images.append(i)
        
        labels = torch.tensor(list(map(float, self.img_labels.iloc[index, 5].split(','))))
        
        if self.target_transform:
            labels = self.target_transform(labels)
        
        return images, labels

class MPII_Tran(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, encoding = 'utf-8', delimiter = ' ')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        sub_path = self.img_labels.iloc[index, 0]
        sub_path = sub_path.replace("\\", "/")
        
        img_path = os.path.join(self.img_dir, sub_path)
        print(img_path)
        exit()

        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(list(map(float, self.img_labels.iloc[index, 7].split(','))))
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class Columbia(Dataset):
    def __init__(self, img_dir, transform = None):
        self.label_file = open(os.path.join(img_dir, "list.txt"), encoding = "utf-8")
        self.img_dir = img_dir
        self.transform = transform
        
        self.img_labels = self.label_file.readlines()
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        print(self.img_labels)
        exit()