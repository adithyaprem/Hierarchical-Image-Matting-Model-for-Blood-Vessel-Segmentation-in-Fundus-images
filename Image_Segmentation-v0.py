# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:34:03 2019

@author: Adithya
"""

import cv2
import os
import glob
import numpy as np
from collections import Counter
from PIL import Image, ImageSequence

class Segmentation:
    def __init__(self,path_img):
        
        self.path_main = os.path.join(os.getcwd(), 'DRIVE dataset')
        self.path_train_imgs = os.path.join(self.path_main, 'training', 'images')
        self.path_mask = os.path.join(self.path_main, 'training', 'mask')
        self.path_out = os.path.join(self.path_main, 'Green_Channel_Images')
        self.path_B = os.path.join(self.path_main, 'Background')
        self.path_U = os.path.join(self.path_main, 'Unknown')
        self.path_V1 = os.path.join(self.path_main, 'Vessel Region V1')
        
    def preproc(self):
                
        files = glob.glob(os.path.join(self.path_train_imgs, '*.tif'))
        
        for file in files:
            img = cv2.imread(file)
            #cv2.imshow('Actual Image', img)
            green_img = np.zeros_like(img)
            green_img = img[:,:, 1]
            cv2.imwrite(os.path.join(self.path_out, os.path.basename(file)), green_img)
            #cv2.imshow('Green Channel', green_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        files_g = glob.glob(os.path.join(self.path_out,'*.tif'))
        files_m = glob.glob(os.path.join(self.path_mask,'*.png'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        path_tophat = os.path.join(self.path_main, 'Tophat_images')
        for file in files_g:
            img = cv2.imread(file,0)
            
            contrast_enhanced_green_fundus = clahe.apply(img)
            imgC = ~img
            #print(imgC.shape)
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,21))
            self.morph_img = cv2.morphologyEx(imgC, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
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
            cv2.imwrite(os.path.join(self.path_B, os.path.basename(file)), B.astype(np.uint8))
            cv2.imwrite(os.path.join(self.path_U, os.path.basename(file)), U.astype(np.uint8))
            cv2.imwrite(os.path.join(self.path_V1, os.path.basename(file)), V1.astype(np.uint8))
        return B,U,files_m,files_g
        
    def prepare(self,B,files_m,files_g):
                            
            y = np.ravel(self.morph_img)
#            imgR = B.astype(np.uint8)
            mask = Image.open(files_m[0])
            index = 1
            for frame in ImageSequence.Iterator(mask):
                frame.save("frame%d.png" % index)
                index += 1
            mask = cv2.imread('frame1.png',0)
            result = B * mask
#            imgR = result.astype(np.uint8)
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
            return morph_img,imgC,img,mask,result

path_img = "/home/venseven/adithya/Hierarchical-Image-Matting-Model-for-Blood-Vessel-Segmentation-in-Fundus-images/DRIVE dataset"
vm=Segmentation(path_img)
B,U,files_m,files_g=vm.preproc()
morph_img,imgC,img,mask,result=vm.prepare(B,files_m,files_g)
#
#cv2.imshow('Morphed Image', morph_img)
#            
#cv2.imshow('mask', mask)
#cv2.imshow('Actual', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#    

cv2.imshow('Channel Image', imgC)
cv2.imshow('Morphed Image', img)
#cv2.imshow('Gray Image', gray_img)
cv2.imshow('Background', U.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
