# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:35:44 2019

@author: Asif Towheed
"""

import cv2
from matplotlib import pyplot as plt
from PIL import Image


bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

def matching(desc1, desc2):
    global bf
    matches = bf.match(desc1,desc2)

    matches = sorted(matches, key = lambda x:x.distance)
    
    xmax1 = 0
    ymax1 = 0
    xmin1 = 640
    ymin1 = 480
    
    xmax2 = 0
    ymax2 = 0
    xmin2 = 640
    ymin2 = 480
    
    # the indexes of the 2 pics that have matches
    pic1idxs = []
    pic2idxs = []
    # lists to store the points
    pic1points = []
    pic2points = []
    # the delta x and y
    xdelta = []
    ydelta = []
    
    for i in matches:
        pic1idxs.append(i.trainIdx)
        pic2idxs.append(i.queryIdx)
    
    #find max x,y and min x,y in pic1
    for i in kp:
        if i.pt[0] > xmax1: #max x
            xmax1 = i.pt[0]
        if i.pt[1] > ymax1: #max y
            ymax1 = i.pt[1]
        if i.pt[0] < xmin1: #min x
            xmin1 = i.pt[0]
        if i.pt[1] < ymin1: #min y
            ymin1 = i.pt[1]
    
    #find max x,y and min x,y in pic2
    for i in kp2:
        if i.pt[0] > xmax2:
            xmax2 = i.pt[0] #max x
        if i.pt[1] > ymax2:
            ymax2 = i.pt[1] #max y
        if i.pt[0] < xmin2:
            xmin2 = i.pt[0] #min x
        if i.pt[1] < ymin2:
            ymin2 = i.pt[1] #min y
    
    # print out all the max(s) and min(s)
    print('xmax1', xmax1)
    print('ymax1', ymax1)
    print('xmax2', xmax2)
    print('ymax2', ymax2)
    print('xmin1', xmin1)
    print('ymin1', ymin1)
    print('xmin2', xmin2)
    print('ymin2', ymin2)
    
    # NORMALIZATION
    for i in pic1idxs:
        a = kp[i].pt[0]/(xmax1 - xmin1)
        b = kp[i].pt[1]/(ymax1 - ymin1)
        pic1points.append((a,b))
    
    for i in pic2idxs:
        a = kp2[i].pt[0]/(xmax2 - xmin2)
        b = kp2[i].pt[1]/(ymax2 - ymin2)
        pic2points.append((a,b))
    
    # FOR THE DELTA, SHOULD WE TAKE THE ABSOLUTE VALUE??
    for i in range(len(matches)):
        xdelta.append(pic1points[i][0] - pic2points[i][0])
        ydelta.append(pic1points[i][1] - pic2points[i][1])
        
    return xdelta, ydelta



###############################################################################
i0 = cv2.imread('cap1-en.jpeg')
i1 = cv2.imread('cap2-en.jpeg')


#surf = cv2.xfeatures2d.SURF_create(8000)
#surf = cv2.ORB_create(nfeatures = 10, scaleFactor = 1.8, nlevels = 10)
surf = cv2.ORB_create(nfeatures = 10)

kp, desc1 = surf.detectAndCompute(i0, None)
kp2, desc2 = surf.detectAndCompute(i1, None)

i0 = cv2.drawKeypoints(i0, kp, None, (255,0,0), 4)
i1 = cv2.drawKeypoints(i1, kp2, None, (255,0,0), 4)

#print(len(kp))
#print(len(kp2))
###############################################################################

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(desc1,desc2)

matches = sorted(matches, key = lambda x:x.distance)
#matches[len(matches)-1].distance --> will give the greatest displacement
newmatches = []

# Error correction --> only consider the points with a correct match
for match in matches:
    if abs(kp[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1]) < 20:
        newmatches.append(match)


img3 = cv2.drawMatches(i0,kp,i1,kp2,newmatches[:], None, (0,255,0), flags=2)
img4 = cv2.drawMatches(i0,kp,i1,kp2,matches[:], None, (0,255,0), flags=2)
#print('m', len(matches))

plt.imshow(img3),plt.show()

im = Image.fromarray(img3)
im.save('linesdrawn.jpeg')

xmax1 = 0
ymax1 = 0
xmin1 = 640
ymin1 = 480

xmax2 = 0
ymax2 = 0
xmin2 = 640
ymin2 = 480

# the indexes of the 2 pics that have matches
pic1idxs = []
pic2idxs = []
# lists to store the points
pic1points = []
pic2points = []
# the delta x and y
xdelta = []
ydelta = []

for i in newmatches:
    pic1idxs.append(i.queryIdx)
    pic2idxs.append(i.trainIdx)
    
#print('pic1idxs',pic1idxs)
#print('pic2idxs',pic2idxs)

#find max x,y and min x,y in pic1
for i in kp:
    if i.pt[0] > xmax1: #max x
        xmax1 = i.pt[0]
    if i.pt[1] > ymax1: #max y
        ymax1 = i.pt[1]
    if i.pt[0] < xmin1: #min x
        xmin1 = i.pt[0]
    if i.pt[1] < ymin1: #min y
        ymin1 = i.pt[1]

#find max x,y and min x,y in pic2
for i in kp2:
    if i.pt[0] > xmax2:
        xmax2 = i.pt[0] #max x
    if i.pt[1] > ymax2:
        ymax2 = i.pt[1] #max y
    if i.pt[0] < xmin2:
        xmin2 = i.pt[0] #min x
    if i.pt[1] < ymin2:
        ymin2 = i.pt[1] #min y

# print out all the max(s) and min(s)
print('xmax1', xmax1)
print('ymax1', ymax1)
print('xmax2', xmax2)
print('ymax2', ymax2)
print('xmin1', xmin1)
print('ymin1', ymin1)
print('xmin2', xmin2)
print('ymin2', ymin2)

# NORMALIZATION
#print('pic1idxs',pic1idxs)
for i in pic1idxs:
    a = kp[i].pt[0]/(xmax1 - xmin1)
    b = kp[i].pt[1]/(ymax1 - ymin1)
    pic1points.append((a,b))

#print('pic2idxs',pic2idxs)
#print('kp2', kp2)
#print('len(kp2)', len(kp2))
#print(kp2[i].pt[0])
#print(kp2[i].pt[1])
for i in pic2idxs:
    #print('i', i)
    a = kp2[i].pt[0]/(xmax2 - xmin2)
    b = kp2[i].pt[1]/(ymax2 - ymin2)
    pic2points.append((a,b))

# FOR THE DELTA, SHOULD WE TAKE THE ABSOLUTE VALUE??
for i in range(len(newmatches)):
    xdelta.append(pic1points[i][0] - pic2points[i][0])
    ydelta.append(pic1points[i][1] - pic2points[i][1])

Maxx = max(xdelta)
Minx = min(ydelta)
Maxy = max(xdelta)
Miny = min(ydelta)

print(xdelta)
print(ydelta)

plt.figure(figsize = (10,10))
#plt.hist(xdelta, range = (-5, 5), density=True, rwidth = 0.90, bins=50)
counts, bins, bars = plt.hist(xdelta, range = (-1, 1), density=True, rwidth = 0.90, bins=10)
#plt.hist(xdelta, density=True, rwidth = 0.90, bins=50)
plt.show()
print("================")
import pandas as pd

#plt.ylabel('Probability');

while True:
    cv2.imshow('i0', i0)
    cv2.imshow('i1', i1)
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        cv2.destroyAllWindows()
        break