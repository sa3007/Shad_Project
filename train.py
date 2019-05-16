import json
import numpy as np
import os
import os, random
import torchvision
from torchvision import models, datasets, transforms
import torch
import matplotlib.pyplot as plt
import copy
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import variable
from collections import OrderedDict
from PIL import Image
import time

import utils

argumentparser = argparse.ArgumentParser(description='train.py')

argumentparser.add_argument('data_dir', action="store", default="./flowers/")
argumentparser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
argumentparser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argumentparser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argumentparser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argumentparser.add_argument('--epochs', dest="epochs", action="store", type=int, default=8)
argumentparser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
argumentparser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=1024)

parser = argumentparser.parse_args()
root = parser.data_dir
paths = parser.save_dir
lr = parser.learning_rate
structures = parser.arch
dropout = parser.dropout
hidden_layer = parser.hidden_units
device = parser.gpu
epochs = parser.epochs

def main():
    
    trainloader, v_loader, testloader = utils.load_data(root)
    model, optimizer, criterion = utils.network_construct(structure,dropout,hidden_layer1,lr,device)
    utils.do_deep_learning(model, optimizer, criterion, epochs, 20, trainloader, device)
    utils.save_checkpoint(model,path,structure,hidden_layer1,dropout,lr)
    print("Training Has Been Completed!")

