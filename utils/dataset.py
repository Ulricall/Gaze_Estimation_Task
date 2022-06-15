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
        labels = []

        for idx in tqdm(self.img_labels.index, leave = False):
            face_path = self.img_labels.iloc[idx, 0]
            left_path = self.img_labels.iloc[idx, 1]
            right_path = self.img_labels.iloc[idx, 2]
            
            face_path = face_path.replace("\\", "/")
            left_path = left_path.replace("\\", "/")
            right_path = right_path.replace("\\", "/")
            
            img_path = [os.path.join(self.img_dir, face_path), 
                        os.path.join(self.img_dir, left_path),
                        os.path.join(self.img_dir, right_path)]

            image = []
            for img in img_path:
                i = read_image(img)
                if self.transform:
                    i = self.transform(i)
                image.append(i)
                
            images.append(image)

            label = torch.tensor(list(map(float, self.img_labels.iloc[index, 5].split(','))))
 
            if self.target_transform:
                label = self.target_transform(label)
            
            labels.append(label)
        print(labels[0][0])
        return images, labels

class MPII_Tran(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, encoding = 'utf-8', delimiter = ' ')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        print("Loading MPII Datasets......")
        
        _path = self.img_labels.iloc[idx, 0]
        _path = _path.replace("\\", "/")
        img_path = os.path.join(self.img_dir, _path)
        img = read_image(img_path)
        print(len(img))
        exit()
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(list(map(float, self.img_labels.iloc[idx, 7].split(','))))
        
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

    def __getitem__(self, idx):
        img_dirs = []
        labels = []
        for img_labels in self.img_labels:
            l = img_labels.strip("\n").split(" ")
            img_dir = os.path.join(self.img_dir, l[0])
            img_label = [float(l[2]), float(l[1])]
            
            img_dirs.append(img_dir)
            labels.append(img_label)
        
        print("Loading Columbia Datasets......")
        
        images = []
        for d in tqdm(img_dirs):
            img = read_image(d)
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
            pass
        
        return images, labels