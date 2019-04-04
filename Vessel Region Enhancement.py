# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:35:55 2019

@author: Adithya
"""
import cv2
import glob
import os
from skimage.data import coins
from skimage.morphology import label, remove_small_objects
from skimage.measure import regionprops, find_contours

files_v1 = glob.glob(os.path.join(path_V1,'*.png'))
img = cv2.imread(files_v1[5])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, contours,hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
#V1 = img.astype(np.int32)
V2 = np.zeros_like(img).astype(np.uint8)
a1 = 21 * (584 / 565) * 2
label_img = label(img, connectivity = 2)
props = regionprops(label_img)
count = 0
V2 = remove_small_objects(label_img, min_size = a1)
for prop in props:
    if prop['Area'] > a1:
        count += 1
print(count)
for cnt in contours:
    if cv2.contourArea(cnt) > 10.7:
        cv2.fillPoly(V2, pts = [cnt], color = [255,255,255])
        V2 = img * V2
        #cv2.drawContours(V2, [cnt], 0, (255,255,255), 1)

V2 = V2.astype(np.uint8)       
cv2.imshow('V1', img)
cv2.imshow('V2', V2)
cv2.waitKey(0)
cv2.destroyAllWindows()        