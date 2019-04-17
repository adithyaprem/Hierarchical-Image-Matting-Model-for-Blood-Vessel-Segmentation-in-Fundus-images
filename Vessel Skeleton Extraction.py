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
    C_next = cv2.imread(file,0)
    C_next = clahe.apply(C_next)
    mask = cv2.imread(os.path.join(path_mask, 'frame0.png'), 0)
    #C_next = cv2.cvtColor(C_next, cv2.COLOR_BGR2GRAY)
    #C_next = ~C_next
    #Defining the filter
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.
    W = []
    KSize = [5,9,17,33]
    for scale, KS2 in enumerate(KSize):
        C_curr = C_next
        KS2 = int(KS2/2)
        kernel = np.zeros((1,KSize[scale]), dtype = np.float32)
        kernel[0][0] = C1
        kernel[0][KSize[scale]-1] = C1
        kernel[0][int(KS2/2)] = C2
        kernel[0][int(KSize[scale]/4+KS2)] = C2
        kernel[0][KS2] = C3
        
        k = kernel.T * kernel
        C_next = cv2.filter2D(C_curr, -1, k)
        #C4 = cv2.sepFilter2D(C_next, cv2.CV_32F, kernelX = kernel, kernelY = kernel)
        W.append(C_curr - C_next)
        
    Iiuw = C_next + W[2] + W[3] + W[0] + W[1] 
    Iiuw = Iiuw.astype(np.uint8)
#newfin = cv2.erode(Iiuw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
#Iiuw = ~Iiuw
    cv2.imwrite(os.path.join(path_results, os.path.basename(file)), Iiuw)
for i in range(Iiuw.shape[0]):
    for j in range(Iiuw.shape[1])    
t, th2 = cv2.threshold(Iiuw, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
img = Iiuw * mask 
img = ((Iiuw > (t + 0.155 * 255)) * 255).astype(np.uint8)
img = ~img
img = cv2.bitwise_and(img,img, mask = mask)
cv2.imshow('Filtered Image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = np.ones(img.shape[:2], dtype="uint8") * 255
_, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if((cv2.contourArea(cnt)>759.71)):
        cv2.drawContours(mask,[cnt],-1,0,-1)
        