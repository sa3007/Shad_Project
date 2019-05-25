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

def get_args():

    argumentparser = argparse.ArgumentParser(description='train.py Flower Classification trainer')
    argumentparser.add_argument('--gpu', type=bool, default=False, help='Use GPU in Training')
    argumentparser.add_argument('--arch', type=str, default='densenet', help='architecture [available: densenet, vgg 16]', required=True)
    argumentparser.add_argument('--lr', type=float, default=0.001, help="set hyperparameter: learning rate (0.01)")
    argumentparser.add_argument('--hidden_units', type=int, default=100, help="set the directory to save checkpoints")
    argumentparser.add_argument('--epochs', type=int, default=3, help="set hyperparameter: epochs (20)")
    argumentparser.add_argument('--data_dir', type=str, default='flowers', help="set the directory to save checkpoints")
    argumentparser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')
	
# parse all arguments
    args = argumentparser.parse_args()

# breakpoint: in case there are no command line params
if len( args ) < 1:
    parser.error("Basic usage: python train.py data_directory")	


def main():
    
    trainloader, v_loader, testloader = sutils.load_data(root)
    model, optimizer, criterion = sutils.net_construct(structure,dropout,hidden_layer1,lr,device)
    sutils.deep_learning(model, optimizer, criterion, epochs, 20, trainloader, device)
    sutils.save_checkpoint(model,path,structure,hidden_layer1,dropout,lr)
    print("Hurray Completed The training!")


if __name__== "__main__":
    main()
