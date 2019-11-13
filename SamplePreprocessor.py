from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import imutils


def preprocessor(filePath, imgSize, rotation=True):
    # print(filePath)
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    random_rotation = np.random.choice([True, False], p=[0.05, 0.95])
    if rotation & random_rotation:
        rotate_angle = np.random.randint(-15, 15)
        img = imutils.rotate_bound(img, rotate_angle)

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))

    img = cv2.resize(img, newSize)

    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    target = target / 255.0
    target = target.T
    target = np.expand_dims(target, axis=2)
    return target
