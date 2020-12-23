import os
import dlib
import cv2

DATA_PATH = "data/before"
OUTPUT_PATH = "cropped/before"

count_im = 1
face_detector = dlib.get_frontal_face_detector()
im_list = [f in os.listdir(DATA_PATH) if '.png' in f]
for im_name in im_list:
    img = cv2.imread(os.path.join(DATA_PATH, im_name))
    faces = face_detector(img)
    for f in faces:
        crop = img[f.top():f.bottom(), f.left():f.right()]
        cv2.imwrite("{0:>05}.png".format(count_im))
        count_im += 1
