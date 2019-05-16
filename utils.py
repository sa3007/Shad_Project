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

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(31),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
cost_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])
test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
cost_data = datasets.ImageFolder(valid_dir, transform=cost_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(cost_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

image_datasets = [train_data, cost_data, test_data]
dataloaders = [train_loader, valid_loader, test_loader]

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def network_construct(structure='vgg19',dropout=0.5, hidden_layer1 = 1024,lr = 0.001,device='gpu'):
model = models.vgg19(pretrained=True)
model

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 1024)),
                                       ('drop', nn.Dropout(p=0.5)),
                                       ('relu', nn.ReLU()),
                                       ('fc2', nn.Linear(1024, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier
model

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

epochs = 8
steps = 0
cuda = torch.cuda.is_available()

if cuda:
    model.cuda()
else:
    model.cpu()

running_loss = 0
accuracy = 0

start = time.time()
print('Started the Training')

def do_deep_learning(model, criterion, optimizer, epochs = 8, print_every=20, loader=0, device='gpu'):
for e in range(epochs):
    
    train_mode = 0
    valid_mode = 1
    
    for mode in [train_mode, valid_mode]:
        if mode == train_mode:
            model.train()
        else:
            model.eval()
            
        pass_count = 0
        
        for data in dataloaders[mode]:
            pass_count += 1
            inputs, labels = data
            if cuda == True:
                inputs, labels = variable(inputs.cuda()), variable(labels.cuda())
            else:
                inputs, labels = variable(inputs), variable(labels)
                
            optimizer.zero_grad()
            # Feed-forwarding
            output = model.forward(inputs)
            loss = criterion(output, labels)
            
            # Feed-Backward
            if mode ==train_mode:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
            
        if mode == train_mode:
            print("\nEpoch: {}/{} ".format(e+1, epochs),
                 "\nTraining Loss: {:.4f} ".format(running_loss/pass_count))
        else:
            print("Validation Loss: {:.4f} ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))
        running_loss = 0
                  
time_elapsed = time.time() - start
print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed % 60))

model.eval()
accuracy = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    
pass_count = 0

for data in dataloaders[2]:
    pass_count += 1
    images, labels = data
    
    if cuda == True:
        images, labels = variable(images.cuda()), variable(labels.cuda())
    else:
        images, labels = variable(images), variable(labels)

    output = model.forward(images)
    ps = torch.exp(output).data
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()

print("Testing Accuracy: {:.4f}".format(accuracy/pass_count))

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets[0].class_to_idx
model.cpu()
checkpoint = {'arch': 'vgg19',
              'input_size': 25088,
              'hidden_layer': 1024,
              'output_size' : 102,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'epochs': epochs,
              'classifier' : classifier,
              'batch_size' : 64,
              'learning_rate': 0.01,
              'class_to_idx': model.class_to_idx}
    
torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
# Loading the checkpoint
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer
	
nn_filename = 'checkpoint.pth'

model, optimizer = load_checkpoint(nn_filename)

chkp_model = print(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Opening the image using thumbnail
    im = Image.open(image)
    im.load()
    
    # Resize the image
    if im.size[0] > im.size[1]:
        im.thumbnail((1000, 256))
    else:
        im.thumbnail((256, 10000))
        
    # Crop out the center of the image
    size = im.size
    im = im.crop((size[0]//2 - (224/2), size[1]//2 - (224/2),
                 size[0]//2 + (224/2), size[1]//2 + (224/2)))
    
    #In order to Normalize
    im = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean)/std
    
    im = im.transpose((2, 0, 1))
    
    return im
	
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
	
# Display original picture
im = random.choice(os.listdir('./flowers/test/14/'))
im_path = './flowers/test/14/' + im

with Image.open(im_path) as image:
    plt.imshow(image)
    
torch.Size([5, 224, 224])

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        print("The number of GPU:", torch.cuda.device_count())
        print("Name of Device:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    else:
        model.cpu()
        print("We are using CPU now")
    model.eval()
    
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = variable(image)
    if cuda:
        image = image.cuda()
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
        
    label = []
    
    for i in range(5):
        label.append(ind[index[i]])
            
    return prob, label
	
im = random.choice(os.listdir('./flowers/test/9/'))
image_path = './flowers/test/9/' + im

with Image.open(image_path) as image:
    plt.imshow(image)
	
	
probs, classes = predict(image_path, model)
print(probs)
print(classes)
print([cat_to_name[x] for x in classes])

# TODO: Display an image along with the top 5 classes
prob, classes = predict(im_path, model)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

image = Image.open(im_path)
ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(image)

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
    
y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

plt.show()
