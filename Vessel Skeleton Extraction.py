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
from skimage.exposure import rescale_intensity
from scipy.ndimage import correlate,convolve
import natsort 

path = os.path.join(os.getcwd(), '')
path_mask = os.path.join(os.getcwd(), 'training', 'mask')
path_results = os.path.join(os.getcwd(), 'Binary Images')
files_avail = glob.glob(os.path.join(path, '*.tif'))
masks = os.listdir(path_mask)
masks = natsort.natsorted(masks)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21,21))

def convolve2D(image,kernel):
    (iH, iW) = image.shape
    (kH, kW) = kernel.shape
    pad = (kW - 1) // 2
    img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    w = np.zeros((iH,iW), dtype = "float32")
    output = np.zeros((iH, iW), dtype = "float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            output[y - pad,x - pad] = (roi * kernel).sum()
    w = image - output
    output = rescale_intensity(output, in_range = (0,255))
    output = (output * 255).astype("uint8")
    return output, w


for file,m_ad in zip(files_avail, masks):
    C_curr = cv2.imread(file,0)
    #C_curr = clahe.apply(C_next)
    #mask = cv2.imread(os.path.join(path_mask, 'frame0.png'), 0)
    #C_next = cv2.cvtColor(C_next, cv2.COLOR_BGR2GRAY)
    #C_next = ~C_next
    #Defining the filter
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.
    W = []
    t = True
    KSize = [5,9,17]
    for scale, KS2 in enumerate(KSize):
        
        KS2 = int(KS2/2)
        kernel = np.zeros((1,KSize[scale]), dtype = np.float32)
        kernel[0][0] = C1
        kernel[0][KSize[scale]-1] = C1
        kernel[0][int(KS2/2)] = C2
        kernel[0][int(KSize[scale]/4+KS2)] = C2
        kernel[0][KS2] = C3
        
        k = kernel.T * kernel
        #C_next = cv2.filter2D(C_curr, -1, k)
        #C_next = cv2.sepFilter2D(C_curr, cv2.CV_32F, kernelX = kernel, kernelY = kernel)
        #C_next = convolve(C_curr, k, mode = 'mirror')
        C_next, w = convolve2D(C_curr, k)
        C_curr = C_next
        if(t):
            t = False
            continue
        W.append(w)
    
    # Combining all the wavelet scales
        
    Iiuw = W[0] + W[1]
    mask = cv2.imread(os.path.join(path_mask,m_ad),0)
    
    per_px_inc = 0.22
    epsilon = 0.03
    
    t = np.sort(np.ravel(Iiuw))
    thres = t[int(per_px_inc * len(t)) - 1] + epsilon
        
    bw = Iiuw < thres
    bw = bw.astype(np.uint8) * 255
    fil_bw = cv2.bitwise_and(bw,bw, mask = mask)
    m = np.ones_like(mask) * 255
    m1 = np.ones_like(mask) * 255
    _, contours, _ = cv2.findContours(fil_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < 759.71):
            if(area < 43.7):
                cv2.drawContours(m1,[cnt],-1,0,-1)
            else:
                (x, y, w, h) = cv2.boundingRect(cnt)
                extent = area / float(w * h)
                VRatio = w / float(h)
                if((VRatio >= 2.2)and(extent < 0.25)):
                    cv2.drawContours(m1,[cnt],-1,0,-1)
            cv2.drawContours(m,[cnt],-1,0,-1)
    T3 = cv2.bitwise_and(fil_bw, m, mask = mask) 
    vse = cv2.bitwise_and(fil_bw, m1, mask = mask)
#Iiuw = Iiuw.astype(np.uint8)
#newfin = cv2.erode(Iiuw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
#Iiuw = ~Iiuw
    cv2.imwrite(os.path.join(path_results, os.path.basename(file)), fil_bw)
    cv2.imwrite(os.path.join(os.getcwd(),'T3', os.path.basename(file)), T3)
    cv2.imwrite(os.path.join(os.getcwd(),'Final_VSE', os.path.basename(file)), vse)
    
"""for i in range(Iiuw.shape[0]):
    for j in range(Iiuw.shape[1])    
t, th2 = cv2.threshold(Iiuw, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
img = Iiuw * mask 
img = ((Iiuw > (t + 0.155 * 255)) * 255).astype(np.uint8)
img = ~img
img = cv2.bitwise_and(img,img, mask = mask)
cv2.imshow('T3', T3)
cv2.imshow('T4', T4)
cv2.waitKey(0)
cv2.destroyAllWindows()


mask = np.ones(img.shape[:2], dtype="uint8") * 255"""

for file, m_ad in zip(os.listdir(path_results), masks):
    fil_bw = cv2.imread()
    mask = cv2.imread(os.path.join(path_mask,m_ad),0)

    cv2.imwrite()
