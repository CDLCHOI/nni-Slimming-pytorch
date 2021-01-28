import sys
sys.path.append("..")
import math
import os
import argparse
from numpy import mean

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import vgg19_bn
from torch.optim import SGD
from torchsummary import summary

from nni.compression.torch import SlimPruner

from common.common import print_and_write,print_and_write_with_time
#import sys
#import os
#print(os.path.abspath(sys.modules[SlimPruner.__module__].__file__))

#####################################

# output
batch_size = 64
display = 20

# learning rate
base_lr = 0.001

#训练测试文件
train_file = 'train.txt'
test_file = 'val.txt'
log_file = 'log.txt'

#####################################

def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1
            
            
def test(model):
    model.eval()
    model.cuda()
    test_acc = 0.0
    test_loss = 0
    for i, (images, labels ) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)
        test_loss += nn.CrossEntropyLoss()(outputs, labels).item()
        
    # 正确数量/总数量
    test_acc = test_acc / float(len(test_loader.dataset))
    return test_acc

def train(epoch):

    model.train()
    model.cuda()
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
        loss = nn.CrossEntropyLoss()(outputs,labels)
        
        #存储loss
        loss_queue.append(loss.data.item())
        loss_queue.pop(0)
        
        #反向传播,计算出梯度矩阵
        loss.backward()
        
        #根据梯度更新参数
        optimizer.step()
        
        
        #当前正确个数
        _, prediction = torch.max(outputs.data, 1)
        current_correct_num = torch.sum(prediction == labels.data)
        correct_num_queue.append(current_correct_num)
        correct_num_queue.pop(0)
        
        
        if i!=0 and i%display==0:
            print_and_write_with_time('Epoch {}, Iteration {}, lr = {}'.format(epoch, i, optimizer.param_groups[0]['lr']), log_file)
            print_and_write_with_time('    Training loss ave = {:.3f}'.format(mean(loss_queue)), log_file)
            print_and_write_with_time('    Training accu ave = {:.3f}'.format(sum(correct_num_queue)/(batch_size*display)), log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("XXXXXXXXXXXXXXXXX")
    parser.add_argument('--original_model', type=str, default='../snapshots/vgg_cifar10_sparse_epoch_159.pth')
    parser.add_argument('--pruned_model', type=str, default='../snapshots/pruned_vgg19_cifar10.pth')
    parser.add_argument('--mask_file', type=str, default='../snapshots/mask_vgg19_cifar10.pth')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, shuffle=True,num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, shuffle=False,num_workers=0)



    #定义导入模型
    model = torchvision.models.vgg19_bn(num_classes=10)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, 10)
    model.load_state_dict(torch.load(args.original_model))
    
    #测试基本模型的准确率
    print_and_write_with_time('===Testing Original Model===')
    acc = test(model)
    print_and_write_with_time("Test Accuracy: {}".format(acc), log_file)
    print_and_write_with_time('-------------------------------------------------------------', log_file)
    
    
    
    #裁剪模型，测试准确率.
    optimizer = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    configure_list = [{'sparsity': 0.4,'op_types': ['BatchNorm2d']}]
    pruner = SlimPruner(model, configure_list, optimizer)
    model = pruner.compress()
    print_and_write_with_time('===Testing Pruned Model===')
    acc = test(model)
    print_and_write_with_time("Test Accuracy: {}".format(acc), log_file)
    print_and_write_with_time('-------------------------------------------------------------', log_file)
    
    
    #微调剪过的模型40 epochs，测试准确率
    print_and_write_with_time('=========Fine tuning==========')
    best_top1 = 0
    for epoch in range(40):
        train(epoch)
        test_acc = test()

        print_and_write_with_time("Epoch {}, Test Accuracy: {}".format(epoch, test_acc), log_file)
        print_and_write('-------------------------------------------------------------', log_file)
        
        if test_acc > best_top1:
            best_top1 = test_acc
            pruner.export_model(model_path=args.pruned_model, mask_path=args.mask_file)
            print_and_write('save checkpoint', log_file)
            
    #测试导出的模型
    print_and_write_with_time('=' * 10 + 'Test the export pruned model after fine tune' + '=' * 10)
    new_model = torchvision.models.vgg19_bn(num_classes=10)
    new_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    new_model.classifier = nn.Linear(512, 10)
    new_model.cuda()
    new_model.load_state_dict(torch.load(args.pruned_model))
    test(new_model)
    

