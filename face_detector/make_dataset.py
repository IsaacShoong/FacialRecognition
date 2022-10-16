import cv2
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import os

mtcnn = MTCNN()

def correctDataSet(videoPath,save):
    videoCap = cv2.VideoCapture(videoPath)

    vLen = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    print(vLen)

    for i in range(vLen):
        success, frame = videoCap.read()
        if (success == False):
            pass
        else:
            #cv2.imwrite(rf"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train\Isaac\{i}.jpg", frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
              
    savePath = [save + rf"\\&{i}" + r".jpg" for i in range (len(frames))]

    for frame,path in zip(frames,savePath):
        mtcnn(frame,path)

correctVideoPath = r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\IsaacFace.mp4"
correctSavePath = r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train\Isaac"

def generalDataSet(path,save):
    numFiles = list(range(1026))

    savePath = [save + rf"\\{i}" + r".jpg" for i in range (len(numFiles))]

    for file,toSave in zip(sorted(os.listdir(path)), savePath):
        if (file[-1] == 'g'):
            im = Image.open(path + r"\\" + file)
            mtcnn(im,toSave)

dataPath = r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train\dataSet"
correctDataPath = r"C:\Users\isaac\OneDrive\Documents\FacialRecognition\FacialRecognition\data_set\train\Not_Isaac"

generalDataSet(dataPath,correctDataPath)