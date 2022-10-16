import cv2
from facenet_pytorch import MTCNN
import numpy as np
import torch
import keyboard
from PIL import Image
from torchvision import transforms

class FaceDetect(object):
    def __init__(self,mtcnn, classifier):
        self.mtcnn = mtcnn
        self.classifier = classifier

    def draw(self,frame,box,prob,landMark,colour):
        try:
            for x,y,z in zip(box,prob,landMark):
                x = x.astype('int')

                cv2.rectangle(frame, (x[0],x[1]), (x[2],x[3]), colour, 2)

                cv2.putText(frame, str(round(y,4)),(x[2],x[3]), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2, cv2.LINE_AA)

                z = z.astype('int')
                cv2.circle(frame, tuple(z[0]), 5, colour, -1)
                cv2.circle(frame, tuple(z[1]), 5, colour, -1)
                cv2.circle(frame, tuple(z[2]), 5, colour, -1)
                cv2.circle(frame, tuple(z[3]), 5, colour, -1)
                cv2.circle(frame, tuple(z[4]), 5, colour, -1)
        except Exception as e:
            print(e)
            pass

    def detectROI(self, boxes):
        ROIList = []
        for box in boxes:
            ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIList.append(ROI)
        return ROIList

    def isCorrect(self, face):
        destRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        PILImage = Image.fromarray(destRGB.astype("uint8"), "RGB")
        preProcess = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        preProcessedImage = preProcess(PILImage)
        batchT = torch.unsqueeze(preProcessedImage, 0)
        with torch.no_grad():
            out = self.classifier(batchT)
            _, pred = torch.max(out, 1)
        prediction = np.array(pred[0])

        if (prediction == 0):
            return "Good"
        else:
            return "Bad"

    def run(self):
        faceCapture = cv2.VideoCapture(0)
        colour = (0,0,255)

        while True:
            output, frame = faceCapture.read()
            frame = cv2.flip(frame,1)

            try:
                box, prob, landMark = self.mtcnn.detect(frame,True)
                #self.draw(frame,box,prob,landMark,colour)
                
                try:
                    ROIs = self.detectROI(box)
    
                    for ROI in ROIs:
                        (startY,endY,startX,endX) = ROI
                        face = frame[startY:endY, startX:endX]

                        pred = self.isCorrect(face)
                        if (pred == "Good"):
                            colour = (0,255,0)
                            self.draw(frame,box,prob,landMark,colour)
                        else:
                            colour = (0,0,255)
                            self.draw(frame,box,prob,landMark,colour)
                    
                except Exception as e:
                    print(e)
                    pass

                cv2.imshow("Face Detector", frame)
            except Exception as e:
                print(e)
                print("Problem 2")
                pass

            if (cv2.waitKey(1) and keyboard.is_pressed("Esc")): #Break loop
                break

        faceCapture.release()
        cv2.destroyAllWindows()

"""
mtcnn = MTCNN()
face = FaceDetect(mtcnn)
face.run()
"""