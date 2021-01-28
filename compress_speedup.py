import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
from torchvision import datasets, transforms
import time
import sys
sys.path.append(r'D:\行为识别\pytorch-cifar-master')

from nni.compression.pytorch import ModelSpeedup, apply_compression_results

def test(model):
    model.eval()
    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=32, shuffle=False,num_workers=0)
    
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

model = torchvision.models.vgg19_bn(num_classes=10)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.classifier = nn.Linear(512, 10)
model.load_state_dict(torch.load(r'prune/pruned_vgg19_cifar10.pth'))
model.cuda()

print(test(model))

dummy_input = torch.ones([64,3,32,32]).cuda()

# mask
use_mask_out = use_speedup_out = None
apply_compression_results(model, 'prune/mask_vgg19_cifar10.pth')
start = time.time()
for _ in range(32):
    use_mask_out = model(dummy_input)
print('elapsed time when use mask: ', time.time() - start)
print(test(model))


# speedup
m_speedup = ModelSpeedup(model, dummy_input, 'prune/mask_vgg19_cifar10.pth')
m_speedup.speedup_model()
start = time.time()
for _ in range(32):
    use_speedup_out = model(dummy_input)
print('elapsed time when use speedup: ', time.time() - start)

print(test(model))

# 确定两输出是一致的
if torch.allclose(use_mask_out, use_speedup_out, atol=1e-07):
    print('the outputs from use_mask and use_speedup are the same')       


# 保存模型
trace_model = torch.jit.trace(model,torch.ones(1,3,32,32).cuda())
trace_model.save('trace_vgg_cifar10.pt')
print(test(trace_model))
    









