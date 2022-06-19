import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_ex = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 12, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 24 * 24, 1200)
        self.fc2 = nn.Linear(1200, 50)
        self.fc3 = nn.Linear(50, 3)
        
    def forward(self, face, left, right):
        x_f = self.pool1(F.relu(self.conv1(face)))
        x_l = self.pool1(F.relu(self.conv1_ex(left)))
        x_r = self.pool1(F.relu(self.conv1_ex(right)))
        
        x_f = self.pool2(F.relu(self.conv2(x_f)))
        x_l = self.pool2(F.relu(self.conv2(x_l)))
        x_r = self.pool2(F.relu(self.conv2(x_r)))
        
        x_f = self.pool3(F.relu(self.conv3(x_f)))
        x_l = self.pool3(F.relu(self.conv3(x_l)))
        x_r = self.pool3(F.relu(self.conv3(x_r)))
        
        x = x_f + (x_l + x_r) / 2
        
        x = x.view(-1, 12 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x / torch.norm(x, p = 2, dim = 1).unsqueeze(1).repeat(1, 3)
        
        return x

class TranGazeNet(nn.Module):
    def __init__(self):
        super(TranGazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 12, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 24 * 24, 1200)
        self.fc2 = nn.Linear(1200, 50)
        self.fc3 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 12 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x / torch.norm(x, p = 2, dim = 1).unsqueeze(1).repeat(1, 3)

        return x
    
class TranGazeNet_2(nn.Module):
    def __init__(self):
        super(TranGazeNet_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 12, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 24 * 24, 1200)
        self.fc2 = nn.Linear(1200, 50)
        self.fc3 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 12 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x / torch.norm(x, p = 2, dim = 1).unsqueeze(1).repeat(1, 2)

        return x