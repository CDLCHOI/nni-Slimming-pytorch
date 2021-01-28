# nni-Slimming-pytorch

pytorch 1.7.1+cu110
torchvision 0.8.2+cu110
nni 2.0

model is the VGG19_bn without the two 4096 fc layer from torchvision

1.train with sparsity from scratch
epoch and learning_rate:
1-80 0.001
81-120 0.0001
121-160 0.00001
# python main_train.py --sparsity_lambda 0.0001     


# python prune_slimming.py


# python compress_speedup.py
