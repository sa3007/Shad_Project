import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import time
import argparse

import sutils

argumentparser = argparse.ArgumentParser(description='Train.py')


argumentparser.add_argument('data_dir', action="store", default="./flowers/")
argumentparser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
argumentparser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argumentparser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argumentparser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argumentparser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
argumentparser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
argumentparser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

parser = argumentparser.parse_args()
root = parser.data_dir
path = parser.save_dir
lr = parser.learning_rate
structure = parser.arch
dropout = parser.dropout
hidden_layer1 = parser.hidden_units
device = parser.gpu
epochs = parser.epochs

def main():
    
    trainloader, v_loader, testloader = sutils.load_data(root)
    model, optimizer, criterion = sutils.net_construct(structure,dropout,hidden_layer1,lr,device)
    sutils.deep_learning(model, optimizer, criterion, epochs, 40, trainloader, device)
    sutils.save_checkpoint(model,path,structure,hidden_layer1,dropout,lr)
    print("Hurray Completed The training!")


if __name__== "__main__":
    main()

