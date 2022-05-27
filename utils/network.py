import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_ex = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1200)
        self.fc2 = nn.Linear(1200, 50)
        self.fc3 = nn.Linear(50, 3)
        
    def forward(self, face, left, right):
        x_f = self.pool(F.relu(self.conv1(face)))
        x_l = self.pool(F.relu(self.conv1_ex(left)))
        x_r = self.pool(F.relu(self.conv1_ex(right)))
        
        x_f = self.pool(F.relu(self.conv2(x_f)))
        x_l = self.pool(F.relu(self.conv2(x_l)))
        x_r = self.pool(F.relu(self.conv2(x_r)))
        
        x = x_f + (x_l + x_r) / 2
        
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x / torch.norm(x, p = 2, dim = 1).unsqueeze(1).repeat(1, 3)
        return x
