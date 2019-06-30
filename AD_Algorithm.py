# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:34:52 2019

@author: Asif Towheed
"""

# =============================================================================
# for filename in os.listdir("C:\Users\Asif Towheed\Documents\Senior_Design\getting"):
#     if filename.endswith(".jpeg"):
#         print(os.path.join("C:\Users\Asif Towheed\Documents\Senior_Design\getting", filename))
#         continue
#     else:
#         continue
# =============================================================================

import os
import glob
from PIL import Image
import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
from skimage import io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pyrealsense2 as rs
import numpy as np
import time
from matplotlib.pyplot import figure

# Getting the background images
#==============================================================================
wbg1 = Image.open("backgroundwasif1.jpeg").convert('L')
wbg2 = Image.open("backgroundwasif2.jpeg").convert('L')
wbg3 = Image.open("backgroundwasif3.jpeg").convert('L')
wbg4 = Image.open("backgroundwasif4.jpeg").convert('L')
mbg1 = Image.open("backgroundmumtaz1.jpeg").convert('L')
mbg2 = Image.open("backgroundmumtaz2.jpeg").convert('L')
mbg3 = Image.open("backgroundmumtaz3.jpeg").convert('L')
mbg4 = Image.open("backgroundmumtaz4.jpeg").convert('L')
wbg1 = np.array(wbg1)
wbg2 = np.array(wbg2)
wbg3 = np.array(wbg3)
wbg4 = np.array(wbg4)
mbg1 = np.array(mbg1)
mbg2 = np.array(mbg2)
mbg3 = np.array(mbg3)
mbg4 = np.array(mbg4)

imsize = Image.open("backgroundwasif1.jpeg").size
width, height = imsize

print(imsize)    
    


i = 0

imagelistw1 = []
numpylistw1 = []
imagelistw2 = []
numpylistw2 = []
imagelistw3 = []
numpylistw3 = []
imagelistw4 = []
numpylistw4 = []

imagelistm1 = []
numpylistm1 = []
imagelistm2 = []
numpylistm2 = []
imagelistm3 = []
numpylistm3 = []
imagelistm4 = []
numpylistm4 = []

accumdiffw1 = np.zeros([height, width], dtype=np.uint8)
accumdiffw2 = np.zeros([height, width], dtype=np.uint8)
accumdiffw3 = np.zeros([height, width], dtype=np.uint8)
accumdiffw4 = np.zeros([height, width], dtype=np.uint8)

accumdiffm1 = np.zeros([height, width], dtype=np.uint8)
accumdiffm2 = np.zeros([height, width], dtype=np.uint8)
accumdiffm3 = np.zeros([height, width], dtype=np.uint8)
accumdiffm4 = np.zeros([height, width], dtype=np.uint8)


#====================================================================================================
def finddiff(im1, im2, n, whichfeature):
    individualdiff2 = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    
    global accumdiffm1
    global accumdiffm2
    global accumdiffm3
    global accumdiffm4
    global accumdiffw1
    global accumdiffw2
    global accumdiffw3
    global accumdiffw4
    
    if whichfeature == "m1":
        accumdiff2 = accumdiffm1
        print("m1")
    elif whichfeature == "m2":
        accumdiff2 = accumdiffm2
        print("m2")
    elif whichfeature == "m3":
        accumdiff2 = accumdiffm3
        print("m3")
    elif whichfeature == "m4":
        accumdiff2 = accumdiffm4
        print("m4")
    elif whichfeature == "w1":
        accumdiff2 = accumdiffw1
        print("w1")
    elif whichfeature == "w2":
        accumdiff2 = accumdiffw2
        print("w2")
    elif whichfeature == "w3":
        accumdiff2 = accumdiffw3
        print("w3")
    elif whichfeature == "w4":
        accumdiff2 = accumdiffw4
        print("w4")
    
    sumofdiff = 0
    
    for w in range(width):
        for h in range(height):
            absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
            #print("PRINTING", int(im1[h,w]))
            if absdiff < 30:
                absdiff = 0
            sumofdiff += absdiff
            individualdiff2[h,w] = absdiff
    
    newthresh = sumofdiff/(width*height)
    
    for w in range(width):
        for h in range(height):
            if individualdiff2[h,w] < newthresh:
                individualdiff2[h,w] = 0
            else:
                accumdiff2[h,w] += 10
    
    if whichfeature == "m1":
        accumdiffm1 = accumdiff2
        print("m1")
    elif whichfeature == "m2":
        accumdiffm2 = accumdiff2
        print("m2")
    elif whichfeature == "m3":
        accumdiffm3 = accumdiff2
        print("m3")
    elif whichfeature == "m4":
        accumdiffm4 = accumdiff2
        print("m4")
    elif whichfeature == "w1":
        accumdiffw1 = accumdiff2
        print("w1")
    elif whichfeature == "w2":
        accumdiffw2 = accumdiff2
        print("w2")
    elif whichfeature == "w3":
        accumdiffw3 = accumdiff2
        print("w3")
    elif whichfeature == "w4":
        accumdiffw4 = accumdiff2
        print("w4")
        
    print("SAVING")
    newimage = Image.fromarray(individualdiff2)
    newimage.save("IND-IMAGE-{}.jpeg".format(n))
            
# =============================================================================
#     newimage = Image.fromarray(individualdiff2)
#     newimage.save("IND-IMAGE-{}.jpeg".format(n))
# #====================================================================================================
# =============================================================================

print("-- getting images started")
for filepath in glob.iglob(r'C:\Users\Asif Towheed\Documents\Senior_Design\getting/*.jpeg'):
    path = r'C:\Users\Asif Towheed\Documents\Senior_Design\getting/*.jpeg'
#    print(filepath)
#    print(path)
    imgname = filepath[54:]
    if "mumtaz1" in imgname and "background" not in imgname:
        imagelistm1.append(Image.open(imgname).convert("L"))
        numpylistm1.append(np.array(Image.open(imgname).convert("L")))
    elif "mumtaz2" in imgname and "background" not in imgname:
        imagelistm2.append(Image.open(imgname).convert("L"))
        numpylistm2.append(np.array(Image.open(imgname).convert("L")))
    elif "mumtaz3" in imgname and "background" not in imgname:
        imagelistm3.append(Image.open(imgname).convert("L"))
        numpylistm3.append(np.array(Image.open(imgname).convert("L")))
    elif "mumtaz4" in imgname and "background" not in imgname:
        imagelistm4.append(Image.open(imgname).convert("L"))
        numpylistm4.append(np.array(Image.open(imgname).convert("L")))
    elif "wasif1" in imgname and "background" not in imgname:
        imagelistw1.append(Image.open(imgname).convert("L"))
        numpylistw1.append(np.array(Image.open(imgname).convert("L")))
    elif "wasif2" in imgname and "background" not in imgname:
        imagelistw2.append(Image.open(imgname).convert("L"))
        numpylistw2.append(np.array(Image.open(imgname).convert("L")))
    elif "wasif3" in imgname and "background" not in imgname:
        imagelistw3.append(Image.open(imgname).convert("L"))
        numpylistw3.append(np.array(Image.open(imgname).convert("L")))
    elif "wasif4" in imgname and "background" not in imgname:
        imagelistw4.append(Image.open(imgname).convert("L"))
        numpylistw4.append(np.array(Image.open(imgname).convert("L")))
print("== getting images ended")
        
print("-- diff w1 started")
for i in range(len(numpylistw1) - 1):
    finddiff(numpylistw1[i], numpylistw1[i+1], i, "w1")
print("== diff w1 ended")


print("-- diff w2 started")
for i in range(len(numpylistw2) - 1):
    finddiff(numpylistw2[i], numpylistw2[i+1], i, "w2")
print("== diff w2 ended")


print("-- diff w3 started")
for i in range(len(numpylistw3) - 1):
    finddiff(numpylistw3[i], numpylistw3[i+1], i, "w3")
print("== diff w3 ended")


print("-- diff w4 started")
for i in range(len(numpylistw4) - 1):
    finddiff(numpylistw4[i], numpylistw4[i+1], i, "w4")
print("== diff w4 ended")


print("-- diff m1 started")
for i in range(len(numpylistm1) - 1):
    finddiff(numpylistm1[i], numpylistm1[i+1], i, "m1")
print("== diff m1 ended")


print("-- diff m2 started")
for i in range(len(numpylistm2) - 1):
    finddiff(numpylistm2[i], numpylistm2[i+1], i, "m2")
print("== diff m2 ended")


print("-- diff m3 started")
for i in range(len(numpylistm3) - 1):
    finddiff(numpylistm3[i], numpylistm3[i+1], i, "m3")
print("== diff m3 ended")


print("-- diff m4 started")
for i in range(len(numpylistm4) - 1):
    finddiff(numpylistm4[i], numpylistm4[i+1], i, "m4")
print("== diff m4 ended")


# Getting the DCT
#==============================================================================
print("-- dct m1 started")
dctm1 = dct( dct( accumdiffm1, axis=0), axis=1)
print("-- dct m2 started")
dctm2 = dct( dct( accumdiffm2, axis=0), axis=1)
print("-- dct m3 started")
dctm3 = dct( dct( accumdiffm3, axis=0), axis=1)
print("-- dct m4 started")
dctm4 = dct( dct( accumdiffm4, axis=0), axis=1)

print("-- dct w1 started")
dctw1 = dct( dct( accumdiffw1, axis=0), axis=1)
print("-- dct w2 started")
dctw2 = dct( dct( accumdiffw2, axis=0), axis=1)
print("-- dct w3 started")
dctw3 = dct( dct( accumdiffw3, axis=0), axis=1)
print("-- dct w4 started")
dctw4 = dct( dct( accumdiffw4, axis=0), axis=1)


# Getting the feature vector
#==============================================================================
featurevecm1 = []
featurevecm2 = []
featurevecm3 = []
featurevecm4 = []

featurevecw1 = []
featurevecw2 = []
featurevecw3 = []
featurevecw4 = []

print("-- featurevecs started started")
for i in range(10):
    for j in range(10):
        featurevecm1.append(dctm1[i][j])
        featurevecm2.append(dctm2[i][j])
        featurevecm3.append(dctm3[i][j])
        featurevecm4.append(dctm4[i][j])

        featurevecw1.append(dctw1[i][j])
        featurevecw2.append(dctw2[i][j])
        featurevecw3.append(dctw3[i][j])
        featurevecw4.append(dctw4[i][j])


# Feeding the feature vector into the classifier
#==============================================================================
print("-- classifier started started")
featurevecsm1 = np.array(featurevecm1)
featurevecsm1 = featurevecsm1.reshape(1,-1)
featurevecsm2 = np.array(featurevecm2)
featurevecsm2 = featurevecsm2.reshape(1,-1)
featurevecsm3 = np.array(featurevecm3)
featurevecsm3 = featurevecsm3.reshape(1,-1)
featurevecsm4 = np.array(featurevecm4)
featurevecsm4 = featurevecsm4.reshape(1,-1)

featurevecsw1 = np.array(featurevecw1)
featurevecsw1 = featurevecsw1.reshape(1,-1)
featurevecsw2 = np.array(featurevecw2)
featurevecsw2 = featurevecsw2.reshape(1,-1)
featurevecsw3 = np.array(featurevecw3)
featurevecsw3 = featurevecsw3.reshape(1,-1)
featurevecsw4 = np.array(featurevecw4)
featurevecsw4 = featurevecsw4.reshape(1,-1)

featurevecs2 = np.append(featurevecsm1, featurevecsm2, axis = 0)
featurevecs2 = np.append(featurevecs2, featurevecsm3, axis = 0)
featurevecs2 = np.append(featurevecs2, featurevecsw1, axis = 0)
featurevecs2 = np.append(featurevecs2, featurevecsw2, axis = 0)
featurevecs2 = np.append(featurevecs2, featurevecsw3, axis = 0)

print("-- almost there")
clf = svm.SVC(kernel='linear', C = 1.0)     #change to rbf
clf.fit(featurevecs2, ["Mumtaz", "Mumtaz", "Mumtaz", "Wasif", "Wasif", "Wasif"])

y_pred = clf.predict(featurevecsw4)
print(y_pred)







    
    
#    print(imgname)
# =============================================================================
#     if "mumtaz1" in imgname and "background" not in imgname:
#         indimwithoutbg = Image.open(imgname).convert("L")
#         indimwithoutbg2 = np.array(indimwithoutbg)
#         individualdiff = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#         for w in range(width):
#             for h in range(height):
#                 #print(int(mbg1[h,w]))
#                 #print(int(indimwithoutbg2[h,w]))
#                 absdiff = abs((int(indimwithoutbg2[h,w])) - (int(mbg1[h,w])))
#                 if absdiff < 30:
#                     absdiff = 0
#                 else:
#                     absdiff = int(indimwithoutbg2[h,w])
#                 individualdiff[h,w] = absdiff
# 
#                 
#         newmumtaz = Image.fromarray(individualdiff)
#         newmumtaz.save('m1withoutbg{}.jpeg'.format(i))
#         print("Mumtaz1 -- ", imgname)
#     elif "mumtaz2" in imgname and "background" not in imgname:
#         indimwithoutbg = Image.open(imgname).convert("L")
#         indimwithoutbg2 = np.array(indimwithoutbg)
#         individualdiff = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#         for w in range(width):
#             for h in range(height):
#                 #print(int(mbg1[h,w]))
#                 #print(int(indimwithoutbg2[h,w]))
#                 absdiff = abs((int(indimwithoutbg2[h,w])) - (int(mbg1[h,w])))
#                 if absdiff < 30:
#                     absdiff = 0
#                 else:
#                     absdiff = int(indimwithoutbg2[h,w])
#                 individualdiff[h,w] = absdiff
# 
#                 
#         newmumtaz = Image.fromarray(individualdiff)
#         newmumtaz.save('m2withoutbg{}.jpeg'.format(i))
#         print("Mumtaz2 -- ", imgname)
#     elif "mumtaz2" in imgname and "background" not in imgname:
#         indimwithoutbg = Image.open(imgname).convert("L")
#         indimwithoutbg2 = np.array(indimwithoutbg)
#         individualdiff = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#         for w in range(width):
#             for h in range(height):
#                 #print(int(mbg1[h,w]))
#                 #print(int(indimwithoutbg2[h,w]))
#                 absdiff = abs((int(indimwithoutbg2[h,w])) - (int(mbg1[h,w])))
#                 if absdiff < 30:
#                     absdiff = 0
#                 else:
#                     absdiff = int(indimwithoutbg2[h,w])
#                 individualdiff[h,w] = absdiff
# 
#                 
#         newmumtaz = Image.fromarray(individualdiff)
#         newmumtaz.save('m2withoutbg{}.jpeg'.format(i))
#         print("Mumtaz3 -- ", imgname)
#     elif "mumtaz2" in imgname and "background" not in imgname:
#         indimwithoutbg = Image.open(imgname).convert("L")
#         indimwithoutbg2 = np.array(indimwithoutbg)
#         individualdiff = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#         for w in range(width):
#             for h in range(height):
#                 #print(int(mbg1[h,w]))
#                 #print(int(indimwithoutbg2[h,w]))
#                 absdiff = abs((int(indimwithoutbg2[h,w])) - (int(mbg1[h,w])))
#                 if absdiff < 30:
#                     absdiff = 0
#                 else:
#                     absdiff = int(indimwithoutbg2[h,w])
#                 individualdiff[h,w] = absdiff
# 
#                 
#         newmumtaz = Image.fromarray(individualdiff)
#         newmumtaz.save('m2withoutbg{}.jpeg'.format(i))
#         print("Mumtaz -- ", imgname)
#     elif "mumtaz2" in imgname and "background" not in imgname:
#         indimwithoutbg = Image.open(imgname).convert("L")
#         indimwithoutbg2 = np.array(indimwithoutbg)
#         individualdiff = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#         for w in range(width):
#             for h in range(height):
#                 #print(int(mbg1[h,w]))
#                 #print(int(indimwithoutbg2[h,w]))
#                 absdiff = abs((int(indimwithoutbg2[h,w])) - (int(mbg1[h,w])))
#                 if absdiff < 30:
#                     absdiff = 0
#                 else:
#                     absdiff = int(indimwithoutbg2[h,w])
#                 individualdiff[h,w] = absdiff
# 
#                 
#         newmumtaz = Image.fromarray(individualdiff)
#         newmumtaz.save('m2withoutbg{}.jpeg'.format(i))
#         print("Mumtaz -- ", imgname)
# =============================================================================

        
    #indim = Image.open(imgname)
    #indim.save('mumtaz4-{}.jpeg'.format(i))
