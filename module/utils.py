import cv2
import numpy as np
import os

# đọc và tiền xử lý ảnh
def load_images(folder, img_size=(80,80)):
    img,labels = [], []
    for label_name in os.listdir(folder):
        subdir = os.path.join(folder, label_name)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            fpath = os.path.join(subdir, fname)
            img = cv2.imread(fpath,cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img,img_size).flatten()
            img.append(img)
            labels.append(label_name)
        return np.array(img), np.array(labels)