# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:42:13 2021

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
"""
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from torch.utils.data import TensorDataset, DataLoader

# for jupyter notebook only
# for others, use
# from tqdm import tqdm
from tqdm.auto import tqdm

class VGG1132(nn.Module):

    def __init__(self, in_channels):

        super(VGG1132, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_4 = nn.BatchNorm2d(256)
        
        self.fc4 = nn.Linear(4096, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 10)

    def forward(self, x, open_dropout = False):

        # Functions
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if open_dropout:
            dropout_conv = nn.Dropout(0.2)
            dropout_fc = nn.Dropout(0.2)
        else:
            dropout_conv = nn.Dropout(0)
            dropout_fc = nn.Dropout(0)

        # Conv Block #1
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # Conv Block #2
        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # Conv Block #3
        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # FC Block
        h = h.view(h.size(0), -1)
        h = dropout_fc(F.relu(self.fc4(h)))
        h = dropout_fc(F.relu(self.fc5(h)))
        h = self.fc6(h)
        return h
    
# ================= #
# VGG11 Re-designed #
# ================= #

class VGG11(nn.Module):

    def __init__(self, in_channels):

        super(VGG11, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_4 = nn.BatchNorm2d(256)
        
        self.fc4 = nn.Linear(2304, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 10)

    def forward(self, x, open_dropout = True):

        # Functions
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if open_dropout:
            dropout_conv = nn.Dropout(0.2)
            dropout_fc = nn.Dropout(0.2)
        else:
            dropout_conv = nn.Dropout(0.2)
            dropout_fc = nn.Dropout(0.2)

        # Conv Block #1
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # Conv Block #2
        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # Conv Block #3
        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = maxpool(h)
        h = dropout_conv(h)

        # FC Block
        h = h.view(h.size(0), -1)
        h = dropout_fc(F.relu(self.fc4(h)))
        h = dropout_fc(F.relu(self.fc5(h)))
        h = self.fc6(h)
        return h
    
# ============== #
# VGG16 Backbone #
# ============== #
class VGG(nn.Module):
    
    def __init__(self, vgg_structure, in_channels):
        
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(vgg_structure)
        self.classifier = nn.Linear(512, 10)
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    
def VGG16(in_channels):
    if in_channels == 1:
        vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    else:
        vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(vgg_structure, in_channels)
    

# ================= #
# ResNet18 Backbone #
# ================= #
class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    
    def __init__(self, block, num_blocks, in_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels)


class Bottleneck92(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck92, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out
    
class DenseNet92_base(nn.Module):
    
    def __init__(self, block, nblocks, in_channels, growth_rate=12, reduction=0.5, num_classes=10):
        
        super(DenseNet92_base, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
#         print('num_planes', num_planes)
        self.linear = nn.Linear(num_planes*4, num_classes)
        
#         self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
#         num_planes += nblocks[2]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans3 = Transition(num_planes, out_planes)
#         num_planes = out_planes

#         self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
#         num_planes += nblocks[3]*growth_rate

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        if self.in_channels == 1:
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
def DenseNet92(in_channels):
    return DenseNet92_base(Bottleneck92, [16,16,16], in_channels, growth_rate=32)






    
    

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, in_channels, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        if self.in_channels == 1:
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
def DenseNet121(in_channels):
    return DenseNet(Bottleneck, [6,12,24,16], in_channels, growth_rate=32)
    
# ================ #
# Encoding Utility #
# ================ #
def train_epoch(device, epoch, epoch_max, net, trainloader, criterion, optimizer):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    loop = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in loop:
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_description(f'Epoch [{epoch+1}/{epoch_max}]')
        loop.set_postfix(loss=loss.item(), acc=correct / total)
     
    
    
def save_encoding_result(file_path, dataloader, net, device):
    output_x_np_list = []
    output_y_np_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs_np = torch.Tensor.cpu(outputs).detach().numpy()
        output_x_np_list.append(outputs_np)
        output_y_np_list.append(targets.numpy())

    encode_x_np = np.vstack(output_x_np_list)
    encode_y_np = np.hstack(output_y_np_list)
    encode_y_np = encode_y_np.reshape(encode_y_np.shape[0], 1)

    encode_XY = np.hstack([encode_x_np, encode_y_np])

    df = pd.DataFrame(encode_XY)
    df.to_csv(file_path + '.csv.gz', compression='gzip', index=False, header=None)
    del df
    
    
def image_data_to_dataloader(X_np, Y_np, batch_size):
    
    tensor_x = torch.Tensor(X_np / 255.0)
    tensor_x = tensor_x.permute(0, 3, 1, 2)
    tensor_y = torch.Tensor(Y_np.astype(int))
    tensor_y = tensor_y.long()

    # print(tensor_x.shape, tensor_y.shape)
    while X_np.shape[0]%batch_size==1:
        # print('  ', batch_size)
        batch_size += 1
    # print(X_np.shape[0]%batch_size)
    dataset_normFal = TensorDataset(tensor_x, tensor_y)
    dataloader_normFal = DataLoader(dataset_normFal,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2)
    return dataloader_normFal


def encoding(dataloader, net, device):
    output_x_np_list = []
    output_y_np_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs_np = torch.Tensor.cpu(outputs).detach().numpy()
        output_x_np_list.append(outputs_np)
        output_y_np_list.append(targets.numpy())

    encode_x_np = np.vstack(output_x_np_list)
    encode_y_np = np.hstack(output_y_np_list)
    encode_y_np = encode_y_np.reshape(encode_y_np.shape[0], 1)

    encode_XY = np.hstack([encode_x_np, encode_y_np])
    return encode_XY


def set_random_seed(r_seed, device):
    os.environ['PYTHONHASHSEED'] = str(r_seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    tf.random.set_seed(r_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(r_seed)
        torch.cuda.manual_seed_all(r_seed)
        torch.cuda.random.manual_seed(r_seed)
        
        
def build_encoder(dataset_X_dict, dataset_Y_dict, backbone, device, r_seed, epoch_max, dataset_name, lr=0.05):
    
    channel = dataset_X_dict['kn_tr'][0].shape[-1]
    print('channel', channel)
    encoder_list = []
    for i in range(len(dataset_X_dict['kn_tr'])):

        trDataset = image_data_to_dataloader(dataset_X_dict['kn_tr'][i], dataset_Y_dict['kn_tr'][i], 128)
        teDataset = image_data_to_dataloader(dataset_X_dict['kn_te'][i], dataset_Y_dict['kn_te'][i], 128)
        
        encoder_path = 'encoding/checkpoint/'
        encoder_path +=  f'{dataset_name}_{backbone.__name__}_'
        encoder_path += '_'.join([str(num) for num in np.unique(dataset_Y_dict['kn_tr'][i])])
        encoder_path += '.pth'

        net = backbone(channel)
        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        
        if not os.path.exists(encoder_path):

            set_random_seed(r_seed, device)
            optimizer = torch.optim.SGD(net.parameters(),
                                  lr=lr,
                                  momentum=0.9,
                                  weight_decay=5e-4)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            # Build Encoder
            for epoch in range(0, epoch_max):
                train_epoch(device, epoch, epoch_max, net, trDataset, criterion, optimizer)
                scheduler.step()

            torch.save(net, encoder_path)
        else:
            net = torch.load(encoder_path)
        
        net.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(teDataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(acc, loss)
        
        encoder_list.append(net)
        
    return encoder_list
            