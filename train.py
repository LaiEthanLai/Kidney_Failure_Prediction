import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from model.net import Net
from dataset.dataset import Kidney_Dataset
import argparse
import pandas as pd 

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classnum', '--c', type=int, default=2)
    parser.add_argument('--data', '--d', type=str, default='preprocessed_data/tsaoapp_with_filter_data.csv')
    parser.add_argument('--label', '--l', type=str, default='preprocessed_data/tsaoapp_with_filter_label.csv')
    parser.add_argument('--epoch', '--e', type=int, default=100)
    parser.add_argument('--lr_rate', '--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--batch-size', '--b', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimzer', '--opt', type=string, default='Adam')
    parser.add_argument('--save-loss', type=str, default='runs/loss/out.csv')
    parser.add_argument('--save-acc', type=str, default='runs/acc/out.csv')

    opt = parser.parse_args()
    return opt

def train(opt):
    print(opt)
    #####################
    # Hyperparameter

    warm_up_epoch = opt.warmup
    EPOCH = opt.epoch + warm_up_epoch
    Batch_size = opt.batch_size
    Learning_rate = opt.lr_rate
    init_lr = 1e-5
    data_path = opt.data
    lable_path = opt.label

    #####################



    device = torch.device(opt.device)
    dataset = Kidney_Dataset(data_path, lable_path)
    train_set_size = int(np.round(0.9 * dataset.__len__()))
    valid_set_size = dataset.__len__() - train_set_size

    print(f'train_set_size = {train_set_size}')
    print(f'valid_set_size = {valid_set_size}')

    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])

    trainloader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
    validloader = DataLoader(val_set, batch_size=Batch_size)

    train_batch_num = train_set_size / Batch_size
    valid_batch_num = valid_set_size / Batch_size

    net = Net(opt.classnum)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam([ {'params':net.parameters(), 'lr':init_lr}], lr=Learning_rate)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW([ {'params':net.parameters(), 'lr':init_lr}], lr=Learning_rate)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD({'params':net.parameters(), 'lr':init_lr}, lr=Learning_rate)

    loss_pack = []
    val_acc = []

    for epoch in range(1, EPOCH+1):
        running_loss = 0.0
        val_loss = 0.0
        val_image_num = 0
        val_hit = 0
        train_hit = 0
        train_image_num = 0
        a = 0
        if epoch <= warm_up_epoch:
            optimizer.param_groups[0]['lr']= (Learning_rate - init_lr)/(warm_up_epoch-1) * (epoch-1) + init_lr
        elif epoch == 75:
            optimizer.param_groups[0]['lr']*= 0.05
        elif epoch == 100:
            optimizer.param_groups[0]['lr']*= 0.01

        print(optimizer.param_groups[0]['lr'])
        for i, data in enumerate(trainloader):
            inputs, labels = data['Data'].to(device), data['Label'].to(device)

            optimizer.zero_grad()
            outputs = net(inputs.float())
            
            labels = labels.view(labels.shape[0])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()

            train_image_num += inputs.shape[0]
            for i in range(len(outputs)):
                if(np.argmax(outputs[i].cpu().detach().numpy())==labels[i]):
                    train_hit += 1
            
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data['Data'].to(device), data['Label'].to(device)

                outputs = net(inputs.float())
                labels = labels.view(labels.shape[0])
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_image_num += inputs.shape[0]

                for i in range(len(outputs)):
                    if(np.argmax(outputs[i].cpu().detach().numpy())==labels[i]):
                        val_hit += 1

        #lr_sche1.step(epoch)
        #lr_sche2.step(epoch)
        print('Epoch:%3d'%epoch, '|Train Loss:%8.4f'%(running_loss/train_batch_num), '|Train Acc:%3.4f'%(train_hit/(train_image_num)*100.0))
        print('Epoch:%3d'%epoch, '|Valid Loss:%8.4f'%(val_loss/valid_batch_num), '|Valid Acc:%3.4f'%(val_hit/(val_image_num)*100.0))
        val_acc.append((val_hit/(val_image_num)*100.0))
        loss_pack.append(val_loss/valid_batch_num) 

    
    loss_df = pd.DataFrame(data=loss_pack)
    val_acc_df = pd.DataFrame(data=val_acc)
    loss_df.to_csv(opt.save_loss, index_label=False, index = False,header = False)
    val_acc_df.to_csv(opt.save_acc, index_label=False, index = False,header = False)


if __name__ == "__main__":
    opt = parsing()
    train(opt)