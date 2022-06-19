import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import dataset, network, angle, log

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.ToTensor()

    MPII_data = []
    for i in range(10, 15):
        data_tmp = dataset.MPII_Tran_2(annotations_file = './input/gazeestimate/MPIIFaceGaze/Label/p{0}.label'.format(i),
                                img_dir = './input/gazeestimate/MPIIFaceGaze/Image',
                                transform = transform)
        MPII_data.append(data_tmp)
        
    Columbia_data = dataset.Columbia(img_dir = './input/gazeestimate/Columbia_Gaze_Data_Set',
                                    transform = transform)
    
    test_loader = []
    for i, label in enumerate(MPII_data):
        test_loader.append(DataLoader(label, batch_size = 64, shuffle = True))
    
    train_loader = DataLoader(Columbia_data, batch_size = 64, shuffle = True)

    model = network.TranGazeNet_2()
    optimizer = optim.Adam(model.parameters(),lr = 0.25)
    #scheduler=stepLR(optimizer,step_size=lr_patience,gamma=lr_decay_factor)
    criterion = nn.L1Loss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using {0} device".format(device))
    
    model.train()
    for times in tqdm(range(5), leave = False):
        for t_data, t_targets in train_loader:
            img = t_data.cuda()
            targets = t_targets.cuda()
            pred_gaze = model(img)
            loss = criterion(pred_gaze, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    sys.stdout = log.Log(filename = 'outputs_2_Columbia.txt')

    model.eval()
    print("Start testing......")
    for e_loader in tqdm(test_loader, leave = False):
        for e_data, e_targets in e_loader:
            img = e_data.cuda()
            targets = e_targets.cuda()
            pred_gaze = model(img)
            print(angle.angular_error_2d_2(targets, pred_gaze))