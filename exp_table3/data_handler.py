from scipy.io import loadmat
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (mnisTr_X, mnisTr_Y), (mnisTe_X, mnisTe_Y) = mnist.load_data()
    mnisTr_X, mnisTe_X = mnisTr_X / 255.0, mnisTe_X / 255.0
    trX = mnisTr_X.reshape(-1, 28, 28, 1)
    trY = mnisTr_Y
    teX = mnisTe_X.reshape(-1, 28, 28, 1)
    teY = mnisTe_Y
    return trX, trY, teX, teY


def load_svhn():
    images_tr = loadmat('data/1_raw/SVHN/train_32x32.mat')
    images_te = loadmat('data/1_raw/SVHN/test_32x32.mat')
    trX = np.moveaxis(images_tr['X'], -1, 0)/255.0
    trY = np.squeeze(images_tr['y'])
    teX = np.moveaxis(images_te['X'], -1, 0)/255.0
    teY = np.squeeze(images_te['y'])
    return trX, trY, teX, teY


def load_cifar10():
    (ci10Tr_X, ci10Tr_Y), (ci10Te_X, ci10Te_Y) = tf.keras.datasets.cifar10.load_data()
    trX = ci10Tr_X/255.0
    trY = np.squeeze(ci10Tr_Y)
    teX = ci10Te_X/255.0
    teY = np.squeeze(ci10Te_Y)
    return trX, trY, teX, teY


def load_cifar100():
    (ci100Tr_X, ci100Tr_Y), (ci100Te_X, ci100Te_Y) = tf.keras.datasets.cifar100.load_data()
    trX = ci100Tr_X/255.0
    trY = np.squeeze(ci100Tr_Y)
    teX = ci100Te_X/255.0
    teY = np.squeeze(ci100Te_Y)
    return trX, trY, teX, teY


def load_tinyImgNet():

    img_path = './data/1_raw/tinyImgNet/tiny-imagenet-200/val/images/'
    lab_path = './data/1_raw/tinyImgNet/tiny-imagenet-200/val/val_annotations.txt'
    with open(lab_path) as f:
        lab_list = list(f)

    file_list, label_list = [], []
    for line in lab_list:
        file_name, file_lab = line.split('\t')[:2]
        file_list.append(file_name)
        label_list.append(file_lab)
    img_list = []
    for img_name in file_list:
        img = img_path + img_name
        image = Image.open(img)
        image = image.convert(mode='RGB')
        image = image.resize((32, 32)) 
        img_np = np.asarray(image)
        img_list.append(img_np)
    teX = np.array(img_list)
    le = LabelEncoder()
    teY = le.fit_transform(label_list)

    img_tr_path = './data/1_raw/tinyImgNet/tiny-imagenet-200/train/'
    img_tr_path_list = []
    img_tr_name_list = []
    img_list = []
    lab_list = []
    for path, subdirs, files in os.walk(img_tr_path):
        for name in files:
            file_path = os.path.join(path, name)
            if '.JPEG' in name:
                img_tr_path_list.append(file_path)
                img_tr_name_list.append(name)
                image = Image.open(file_path)
                image = image.convert(mode='RGB')
                image = image.resize((32, 32)) 
                img_np = np.asarray(image)
                img_list.append(img_np)
                label, _ = name.split('_')
                lab_list.append(label)
    trX = np.array(img_list)
    trY = le.transform(lab_list)
    
    return trX, trY, teX, teY