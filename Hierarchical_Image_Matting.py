# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:18:31 2019

@author: Adithya
"""

import cv2
import numpy as np
import os
import natsort
import pandas as pd

path_unk = os.path.join(os.getcwd(), 'Unknown')
path_vse = os.path.join(os.getcwd(), 'T3')
path_is = os.path.join(os.getcwd(), 'Vessel Region V2')
path_mask = os.path.join(os.getcwd(), 'training', 'mask')
path_imr = os.path.join(os.getcwd(), 'Tophat_images')

files_vse = natsort.natsorted(os.listdir(path_vse))
files_is = natsort.natsorted(os.listdir(path_is))
files_unk = natsort.natsorted(os.listdir(path_unk))
files_imr = natsort.natsorted(os.listdir(path_imr))

masks = natsort.natsorted(os.listdir(path_mask))

img_is = cv2.imread(os.path.join(path_is, files_is[0]), 0)
img_vse = cv2.imread(os.path.join(path_vse, files_vse[0]), 0)
img_imr = cv2.imread(os.path.join(path_imr, files_imr[0]), 0)

cmb = cv2.bitwise_or(img_is, img_vse)
cmb = cv2.cvtColor(cmb, cv2.COLOR_GRAY2RGB)

img_unk = cv2.imread(os.path.join(path_unk, files_unk[0]), 0)
mask = cv2.imread(os.path.join(path_mask, masks[0]), 0)
img_unk = cv2.bitwise_and(img_unk, img_unk, mask = mask)
red_unk = np.zeros_like(cmb)
red_unk[:,:,2] = img_unk

trimap = cv2.bitwise_or(cmb, red_unk, mask = mask)
t_gray = cv2.bitwise_or(cmb, img_unk, mask = mask)

dist_ves = cv2.distanceTransform(~cmb, cv2.DIST_L2, 3)
dist_back = cv2.distanceTransform(t_gray, cv2.DIST_L2, 3)
dist_unk =  cv2.distanceTransform(~img_unk, cv2.DIST_L2, 3)

unk = []
for i in range(dist_unk.shape[0]):
    for j in range(dist_unk.shape[1]):
        if((dist_unk[i,j] == 0)and(dist_ves[i,j] > 0)):
            unk.append([i,j,dist_ves[i,j],dist_back[i,j]])
unk.sort(key = lambda x:x[2])

grouped = []
D = unk[0][2]
temp = []
for i in unk:
    if i[2] != D:
        grouped.append(temp)
        D = i[2]
        temp = []
    temp.append(i)


    
cv2.imshow('Image Sgmented', img_is)
cv2.imshow('VSE', img_vse)
cv2.imshow('Unknown', red_unk)
cv2.imshow('Trimap', trimap)
cv2.imshow('Combined', cmb)
cv2.imshow('Imr', img_imr)
cv2.waitKey()
cv2.destroyAllWindows()