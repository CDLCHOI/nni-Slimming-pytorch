#Import needed packages
import argparse
import os
import PIL
from numpy import mean
import time
import types

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from dataset import MyDataset
from common.common import print_and_write, current_time
torch.cuda.set_device(0)

import sys
sys.path.append(r'D:\行为识别\pytorch-cifar-master')
from models.resnet import ResNet18

#####################################

# output
batch_size = 64
display = 20
average_loss = 20
snapshot = 1 #epoch
snapshot_prefix = "snapshots/vgg_cifar10_sparse"

# learning rate
base_lr = 0.1
gamma = 0.1
stepsize = [80,120]  #epoch
max_epoch = 160 #epoch

# parameter of SGD
momentum = 0.9
weight_decay = 0.0005
clip_gradients = 40

#训练测试文件
train_file = 'train.txt'
test_file = 'val.txt'
log_file = 'log.txt'

#####################################

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels ) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)
        
    # 正确数量/总数量
    test_acc = test_acc / float(len(test_loader.dataset))
    return test_acc

    
def train(epoch):

    model.train()
    loss_queue = [0]*display #存储
    correct_num_queue = [0]*display
    for i, (images, labels) in enumerate(train_loader):
        # 把训练数据和标签放到cuda
        images = images.cuda()
        labels = labels.cuda()
        
        # 清除累积梯度
        optimizer.zero_grad()
        
        #前向传播
        outputs = model(images)
        
        #计算loss
        loss = loss_fn(outputs,labels)
        
        #存储loss
        loss_queue.append(loss.data.item())
        loss_queue.pop(0)
        
        #反向传播,计算出梯度矩阵
        loss.backward()
        if args.sparsity_lambda:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    #将γ的原本梯度矩阵加上由L1稀疏化引入的梯度
                    m.weight.grad.data.add_(args.sparsity_lambda * torch.sign(m.weight.data))
        #根据梯度更新参数
        optimizer.step()
        
        
        #当前正确个数
        _, prediction = torch.max(outputs.data, 1)
        current_correct_num = torch.sum(prediction == labels.data)
        correct_num_queue.append(current_correct_num)
        correct_num_queue.pop(0)
        
        
        if i!=0 and i%display==0:
            print_and_write(current_time()+'    Epoch {}, Iteration {}, lr = {}'.format(epoch, i, optimizer.param_groups[0]['lr']), log_file)
            #print_and_write(current_time()+'      Training loss cur = {:.3f}'.format(loss.data.item()), log_file)
            print_and_write(current_time()+'      Training loss ave = {:.3f}'.format(mean(loss_queue)), log_file)
            #print_and_write(current_time()+'      Training accu cur = {:.3f}'.format(current_correct_num/batch_size), log_file)
            print_and_write(current_time()+'      Training accu ave = {:.3f}'.format(sum(correct_num_queue)/(batch_size*display)), log_file)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("xxxxxxxxxxxxxxxxxxxx")
    parser.add_argument('--pretrain_weights', type=str, default=None)
    parser.add_argument('--sparsity_lambda', type=float, default=0)
    parser.add_argument('--start_from_epoch', type=int, default=0)
    parser.add_argument('--cifar10', type=str, default=False)
    args = parser.parse_args()
    
    
    #存储目录不存在，就创建
    snapshot_prefix = snapshot_prefix.replace('\\', '/')
    snapshot_folder = '/'.join(snapshot_prefix.split('/')[:-1])
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)
            
    
    #是否cifar10
    if args.cifar10=='1':
        print('is cifar10')
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./data', train=True, download=False,
                           transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=batch_size, shuffle=True,num_workers=0)
        
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=batch_size, shuffle=False,num_workers=0)
    else:
        #训练与测试的数据增强
        train_transformations = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #将0~255的PIL、numpy图片，转成0~1的Tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #将0~1转为-1~1
        ])
        
        test_transformations = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        #创建Dataset类
        train_set = MyDataset(train_file, train_transformations)
        test_set = MyDataset(test_file, test_transformations)
        
        #创建Dataloader
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
        test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0)
    
    
    # 创建模型
    # 1.resnet18
    # model = torchvision.models.resnet18(num_classes=10)
    # model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    # 1.1.resnet18第二种
    # model = ResNet18()
    
    # 1.2.resnet18第三种
    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x) 
    #     return x
    
    # model = torchvision.models.resnet18(num_classes=10)
    # model.conv1 = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1)
    # model._forward_impl = types.MethodType(_forward_impl,model)
    
    # 2.mobilenet_v2
    # model = torchvision.models.mobilenet_v2(num_classes=200)
    
    # 3.vgg19_bn去掉多余全连接
    model = torchvision.models.vgg19_bn(num_classes=10)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, 10)
    
    
    print_and_write(current_time()+'    Epochs:{}, Batch size:{}, Init lr:{}'.format(max_epoch,batch_size,base_lr), log_file)
    
    #如果有指定预训练模型
    if not args.pretrain_weights =='0':
        print_and_write(current_time()+'    Loading from {}'.format(args.pretrain_weights), log_file)
        pretrained_dict = torch.load(args.pretrain_weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #过滤掉不用的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if model_dict[k].shape==v.shape} #过滤掉形状不一样的层
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.cuda()
    
    
    #是否带稀疏训练
    if args.sparsity_lambda:
        print_and_write(current_time()+'    Train with sparsity, λ = {}'.format(args.sparsity_lambda), log_file)
    print_and_write(current_time()+'    A total of {} imgs'.format(len(train_loader.dataset)), log_file)
    
    
    # optimizer = Adam(model.parameters(), lr=base_lr,weight_decay=0.0001)
    optimizer = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    if args.start_from_epoch>0:
        print_and_write(current_time()+'    Resume training from epoch {}'.format(args.start_from_epoch), log_file)
        
    best_acc = 0.0
    for epoch in range(max_epoch):
        
        # step
        if isinstance(stepsize,int):
            if epoch!=0 and epoch % stepsize==0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * gamma
        # multistep
        elif isinstance(stepsize,list):
            if epoch in stepsize:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * gamma
        
        #未达到开始的epoch
        if epoch<args.start_from_epoch:
            continue
        
        train(epoch)
        
        #测试,更新最高准确率，显示测试结果
        test_acc = test()
        if test_acc > best_acc:
            best_acc = test_acc 
        print_and_write(current_time()+"Epoch {}, Test Accuracy: {}".format(epoch, test_acc), log_file)
        print_and_write('save checkpoint', log_file)
        print_and_write('-------------------------------------------------------------', log_file)
    
        #每个epoch完后保存模型
        torch.save(model.state_dict(), "{}_epoch_{}.pth".format(snapshot_prefix, epoch))
        
        print("Checkpoint saved")
    







