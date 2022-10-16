import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy

trainData = np.load(r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train_data.npy", allow_pickle=True)
valData = np.load(r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\val_data.npy", allow_pickle=True)

class Data(Dataset):
  def __init__(self, dataFile):
    self.dataFile = dataFile

  def __len__(self):
    return self.dataFile.shape[0]

  def __getitem__(self, idx):
    image = self.dataFile[idx][0]
    y = int(self.dataFile[idx][1])
    if y == 0:
      label = torch.tensor([1.0,0.0])
    if y == 1:
      label = torch.tensor([0.0,1.0])

    return image, label

# Make data class
trainDataClass = Data(trainData)
valDataClass = Data(valData)

# Make data loader
trainLoader = DataLoader(trainDataClass, batch_size=32, shuffle=True)
valLoader = DataLoader(valDataClass, batch_size=32, shuffle=True)

dataloaders = {'train': trainLoader, 'val': valLoader}
dataset_sizes = {'train': trainData.shape[0], 'val': valData.shape[0]}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Model as feature extractor
modelConv = models.resnet18(pretrained=True)
for param in modelConv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
numFeatures = modelConv.fc.in_features
modelConv.fc = nn.Sequential(nn.Linear(numFeatures, 2), torch.nn.Sigmoid())

criterion = nn.MSELoss()

# Observe that only parameters of final layer are being optimized as opposed to before.
#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizerConv = optim.Adam(modelConv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizerConv, step_size=7, gamma=0.1)

# Testing with one batch

image, label = next(iter(dataloaders['val']))

outputs = modelConv(image)
value, preds = torch.max(outputs, 1)
loss = criterion(outputs, label)

print(loss)

def train_model(model, optimizer,criterion, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    value, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == torch.argmax(labels,1)) #labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_conv = train_model(modelConv,  optimizerConv, criterion,  exp_lr_scheduler,
                       num_epochs=25)

torch.save(model_conv.state_dict(), r'C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\classifier\model-resnet18-2.pth')