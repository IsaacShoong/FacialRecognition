from facenet_pytorch import MTCNN
import Face_Detector 
import os
import torch
from torchvision import models
import torch.nn as nn
import argparse

model = models.resnet18(pretrained = True)
numFeatures = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(numFeatures, 2), torch.nn.Sigmoid())

model.load_state_dict(torch.load(r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\model-resnet18-2.pth"))
model.eval()

mtcnn = MTCNN()
fcd = Face_Detector.FaceDetect(mtcnn, classifier=model)
fcd.run()