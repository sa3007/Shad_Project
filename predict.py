import json
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import argparse

import sutils

argumentparser = argparse.ArgumentParser(description='Predict.py')

argumentparser.add_argument('input', default='./flowers/test/102/image_08012.jpg', nargs='?', action="store", type = str, help='path of image source')
argumentparser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/", help='path of image')
argumentparser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str, help='path of saved model')
argumentparser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='display top k probabilities')
argumentparser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='Path to file containing the categories')
argumentparser.add_argument('--gpu', default="gpu", action="store", dest="gpu", help='Use GPU in Predicting')

parser = argumentparser.parse_args()
path_image = parser.input
number_of_outputs = parser.top_k
device = parser.gpu
category_names = parser.category_names
path = parser.checkpoint
parser = argumentparser.parse_args()

def main():
    model=sutils.load_checkpoint(path)
    with open('category_names', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = sutils.predict(path_image, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Hurrah! Happy to Complete the Predicting!")
    print("Many Thanks to the Mentor assigned to the Program")

if __name__== "__main__":
    main()
    