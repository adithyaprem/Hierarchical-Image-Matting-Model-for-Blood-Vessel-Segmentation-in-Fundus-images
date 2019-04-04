# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:34:03 2019

@author: Adithya
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageSequence
import skimage

#path_img = "G:/Image processing projects/Fundus Blood Vessels Extraction/DRIVE dataset"
print(os.getcwd())

path_main = os.path.join(os.getcwd(), 'DRIVE dataset')
path_train_imgs = os.path.join(path_main, 'training', 'images')
path_mask = os.path.join(path_main, 'training', 'mask')
path_out = os.path.join(path_main, 'Green_Channel_Images')
path_B = os.path.join(path_main, 'Background')
path_U = os.path.join(path_main, 'Unknown')
path_V1 = os.path.join(path_main, 'Vessel Region V1')

files = glob.glob(os.path.join(path_train_imgs, '*.tif'))
print(files)


for file in files:
    img = cv2.imread(file)
    #cv2.imshow('Actual Image', img)
    green_img = np.zeros_like(img)
    green_img = img[:,:, 1]
    cv2.imwrite(os.path.join(path_out, os.path.basename(file)), green_img)
    #cv2.imshow('Green Channel', green_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
files_g = glob.glob(os.path.join(path_out,'*.tif'))
print(files_g)
files_m = glob.glob(os.path.join(path_mask,'*.png'))
print(files_m)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

path_tophat = os.path.join(path_main, 'Tophat_images')
for file in files_g:
    img = cv2.imread(file,0)
    contrast_enhanced_green_fundus = clahe.apply(img)
    #imgC = ~img
    #print(imgC.shape)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,21))
    #morph_img = cv2.morphologyEx(imgC, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)), iterations = 1)	
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)
    cv2.imwrite(os.path.join(path_tophat, os.path.basename(file)), f5)
    Imr = (f5)/max(np.ravel(f5))
    B = (Imr < 0.2)*255
    U = ((Imr > 0.2) * (Imr < 0.35))*255
    V1 = (Imr > 0.35) * 255
    cv2.imwrite(os.path.join(path_B, os.path.basename(file)), B.astype(np.uint8))
    cv2.imwrite(os.path.join(path_U, os.path.basename(file)), U.astype(np.uint8))
    cv2.imwrite(os.path.join(path_V1, os.path.basename(file)), V1.astype(np.uint8))
    
    
y = np.ravel(morph_img)
I_max = max(y)




imgR = B.astype(np.uint8)
mask = Image.open(files_m[0])
index = 1
for frame in ImageSequence.Iterator(mask):
    frame.save("frame%d.png" % index)
    index += 1
mask = cv2.imread('frame1.png',0)

result = B * mask
imgR = result.astype(np.uint8)
cv2.imshow('Morphed Image', morph_img)

cv2.imshow('mask', mask)
cv2.imshow('Actual', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(files_g[0])
imgC = ~img
#print(imgC.shape)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,21))
morph_img = cv2.morphologyEx(imgC, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
gray_img = cv2.cvtColor(morph_img, cv2.COLOR_BGR2GRAY)
_,thres_img = cv2.threshold(morph_img,10,255,cv2.THRESH_BINARY)
mask = (gray_img>30)*255
y = np.ravel(morph_img)
print(max(y))
u = Counter(y)

cv2.imshow('Channel Image', imgC)
cv2.imshow('Morphed Image', img)
#cv2.imshow('Gray Image', gray_img)
cv2.imshow('Background', U.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()