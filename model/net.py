import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, out_feature=2):
        super().__init__()
        
        self.fc1 = nn.Linear(15, 256)
        self.fc2 = nn.Linear(256, 4096)
        self.fc3 = nn.Linear(4096, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, out_feature)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.LeakyReLU()
        self.act5 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(out_feature)
       
        

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.bn1(self.act2(self.fc2(x)))
        x = self.bn2(self.act3(self.fc3(x)))
        x = self.bn3(self.act4(self.fc4(x)))
        x = self.bn4(self.act5(self.fc5(x)))
        x = self.bn5(self.fc6(x))

        return x