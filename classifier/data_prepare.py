import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np
from torchvision import utils
from torchvision import datasets, models, transforms
import os

dataTransforms = {
    "train": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

dataPath = r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set"

imgDataSet = {x: datasets.ImageFolder(os.path.join(dataPath, x), dataTransforms[x]) for x in ['val', 'train']}

trainData = []
for img, lbl in imgDataSet["train"]:
    trainData.append([np.array(img), np.array(lbl)])

trainDatanp = np.array(trainData,dtype=object)
#np.save(r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train_data" + r".npy", trainDatanp)

valData = []
for img, lbl in imgDataSet["val"]:
    trainData.append([np.array(img), np.array(lbl)])

valDatanp = np.array(valData,dtype=object)
np.save(r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\val_data" + r".npy", valDatanp)
