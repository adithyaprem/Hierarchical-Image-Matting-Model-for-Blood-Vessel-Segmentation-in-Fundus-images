# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:18:18 2019

@author: Adithya
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

path = os.path.join(os.getcwd(), 'Green_Channel_Images')
path_mask = os.path.join(os.getcwd(), 'training', 'mask')
path_results = os.path.join(os.getcwd(), 'Wavelet Results')
files_avail = glob.glob(os.path.join(path, '*.tif'))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21,21))
def convolve(kernel, image):
    (iH, iW) = image.shape[:2]
    (kH, )
for file in files_avail:
C_next = cv2.imread(files_avail[0],0)
C_next = clahe.apply(C_next)
mask = cv2.imread(os.path.join(path_mask, 'frame0.png'), 0)
C_next = cv2.cvtColor(C_next, cv2.COLOR_BGR2GRAY)
C_next = ~C_next
#Defining the filter
C1 = 1./16.
C2 = 4./16.
C3 = 6./16.
W = []
KSize = [5,9,17]
for scale in range(3):
    C_curr = C_next
    KS2 = int(KSize[scale]/2)
    kernel = np.zeros((1,KSize[scale]), dtype = np.float32)
    kernel[0][0] = C1
    kernel[0][KSize[scale]-1] = C1
    kernel[0][int(KS2/2)] = C2
    kernel[0][int(KSize[scale]/4+KS2)] = C2
    kernel[0][KS2] = C3
    
    C_next = cv2.filter2D(C_curr, -1, kernel)
    
    W.append(C_curr - C_next)
    
Iiuw = C_next + W[0] + W[1] + W[2]
Iiuw = Iiuw.astype(np.uint8)
Iiuw = ~Iiuw
    cv2.imwrite(os.path.join(path_results, os.path.basename(file)), Iiuw)
for i in range(Iiuw.shape[0]):
    for j in range(Iiuw.shape[1])    
t, th2 = cv2.threshold(Iiuw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
img = Iiuw * mask 
img = ((Iiuw > (t - 0.03 * 255)) * 255).astype(np.uint8)
cv2.imshow('Filtered Image', Iiuw)
cv2.waitKey(0)
cv2.destroyAllWindows()