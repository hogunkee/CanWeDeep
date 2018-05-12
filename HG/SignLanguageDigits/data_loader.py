import cv2
import os
import random

DATAPATH = './Dataset/'

def one_hot(label, total_class):
    tmp = [0 for i in range(total_class)]
    tmp[label] = 1
    return tmp

def data_load(datapath):
    data = []
    label = []
    for classname in os.listdir(datapath):
        classpath = os.path.join(datapath, classname)
        for imagename in os.listdir(classpath):
            imagepath = os.path.join(classpath, imagename)
            img = cv2.imread(imagepath)
            if img.shape == (100,100,3):
                data.append(img)
                label.append(int(classname))

    num_class = max(label) + 1

    class_data = [[] for i in range(num_class)]
    for i in range(len(data)):
        class_data[label[i]].append(data[i])

    data_train = []
    label_train = []
    data_test = []
    label_test = []
    for c in range(num_class):
        data_train += class_data[c][:-10]
        label_train += [c for i in range(len(class_data[c])-10)]
        data_test += class_data[c][-10:]
        label_test += [c for i in range(10)]

    train_dataset = list(zip(data_train, label_train))
    random.shuffle(train_dataset)
    data_train, label_train = zip(*train_dataset)

    label_train = list(map(lambda k: one_hot(k, num_class), label_train))
    label_test = list(map(lambda k: one_hot(k, num_class), label_test))

    return data_train, label_train, data_test, label_test
