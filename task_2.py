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
        #transforms.Normalize(mean=[0.485,0.456,0.406],std=[0,229,0.224,0.225]),
        #transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
        #transforms.RandomGrayscale(p=0.2)
    ])

    target_transform = transforms.ToTensor()

    MPII_data = []
    for i in range(10):
        data_tmp = dataset.MPII_Tran(annotations_file = './input/gazeestimate/MPIIFaceGaze/Label/p0{0}.label'.format(i),
                                    img_dir = './input/gazeestimate/MPIIFaceGaze/Image',
                                    transform = transform)
        MPII_data.append(data_tmp)
        
    Columbia_data = dataset.Columbia(img_dir = './input/gazeestimate/Columbia_Gaze_Data_Set',
                                    transform = transform)
    
    train_loader = []
    for i, label in enumerate(MPII_data):
        train_loader.append(DataLoader(label, batch_size = 64, shuffle = True))
    
    test_loader = DataLoader(Columbia_data, batch_size = 64, shuffle = True)
    
    model = network.TranGazeNet()
    optimizer = optim.Adam(model.parameters(),lr = 0.1)
    #scheduler=stepLR(optimizer,step_size=lr_patience,gamma=lr_decay_factor)
    criterion = nn.L1Loss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    model.train()
    for times in tqdm(range(5)):
        for loader in tqdm(train_loader):
            for data, targets in tqdm(loader):
                img = data[0].cuda()
                targets = targets.cuda()
                pred_gaze = model(img)
                loss = criterion(pred_gaze, targets)
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass
            pass
        pass
    #sys.stdout = log.Log()

    model.eval()
    for loader in tqdm(test_loader):
        for data, targets in tqdm(loader):
            img = data.cuda()
            targets = targets.cuda()
            pred_gaze = model(img)
            print(angle.angular_error(targets, pred_gaze))
            pass
        pass
