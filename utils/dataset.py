import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class GazeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, encoding = 'utf-8', delimiter = ' ')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = []
        for i in range(3):
            _path = self.img_labels.iloc[idx, i]
            _path = _path.replace("\\", "/")
            img_path = os.path.join(self.img_dir, _path)
            img = read_image(img_path)
            if self.transform:
                img = self.transform(img)
            image.append(img)
        
        label = torch.tensor(list(map(float, self.img_labels.iloc[idx, 5].split(','))))
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
