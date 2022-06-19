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

    data = []
    for i in range(15):
        if(i < 10):
            data_tmp = dataset.MPII(annotations_file = './input/gazeestimate/MPIIFaceGaze/Label/p0{0}.label'.format(i),
                                    img_dir = './input/gazeestimate/MPIIFaceGaze/Image',
                                    transform = transform)
        else:
            data_tmp = dataset.MPII(annotations_file = './input/gazeestimate/MPIIFaceGaze/Label/p{0}.label'.format(i),
                                    img_dir = './input/gazeestimate/MPIIFaceGaze/Image',
                                    transform = transform)
        data.append(data_tmp)
    
    train_loader = []
    test_loader = []
    for i, label in enumerate(data):
        if (i < 10):
            train_loader.append(DataLoader(label, batch_size = 64, shuffle = True))
        else:
            test_loader.append(DataLoader(label, batch_size = 64, shuffle = True))
    
    model = network.GazeNet()
    optimizer = optim.Adam(model.parameters(), lr = 0.25)
    criterion = nn.L1Loss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using {0} device".format(device))
    
    model.train()
    print("Start training......")
    for times in tqdm(range(5), leave = False):
        print("Training {0}".format(times + 1))
        for t_loader in tqdm(train_loader, leave = False):
            for t_data, t_targets in t_loader:
                face = t_data[0].cuda()
                left = t_data[1].cuda()
                right = t_data[2].cuda()
                targets = t_targets.cuda()

                pred_gaze = model(face, left, right)
                loss = criterion(pred_gaze,targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    sys.stdout = log.Log(filename = 'outputs_1.txt')

    model.eval()
    print("Start testing......")
    for e_loader in tqdm(test_loader, leave = False):
        for e_data, e_targets in e_loader:
            face = e_data[0].cuda()
            left = e_data[1].cuda()
            right = e_data[2].cuda()
            targets = e_targets.cuda()
            pred_gaze = model(face, left, right)
            print(angle.angular_error_3d(targets, pred_gaze))
