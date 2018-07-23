import cv2
import os

dir='after'
q = []
qsize = []
files = os.listdir(dir)
for file in files:
    img = cv2.imread(os.path.join(dir,file))
    q.append(img)
    qsize.append(img.shape)
