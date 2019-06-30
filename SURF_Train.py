# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:23:23 2019

@author: Asif Towheed
"""

from sklearn import svm
from PIL import Image
import os
import re
import numpy as np
from scipy.fftpack import dct
import cv2
import pickle
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


width, height = (640, 480)



PATH = os.getcwd().replace('\\','/')
PATH_SURF = PATH + '/SURF'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALPATH = ""
INDEX = None

width, height = (640, 480)

XDELTA = []
YDELTA = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []

tInit = time()


class SURF_Histogram:
    
    def __init__(self):
        self.xdelta = []
        self.ydelta = []
        self.range = (-1,1)
        self.bins = 30
        self.density = True
        self.rwidth = 0.90
        self.FVec = []
        
    def addx(self, xdelta):
        self.xdelta += xdelta

    def addy(self, ydelta):
        self.ydelta += ydelta
        
    def resetx(self):
        self.xdelta = []

    def resety(self):
        self.ydelta = []

    def resetfvec(self):
        self.FVec = []
        
    def getfeaturevec(self):
        xcounts, xbins, xbars = plt.hist(self.xdelta, range = self.range, density=self.density, rwidth = self.rwidth, bins=self.bins)
        ycounts, ybins, ybars = plt.hist(self.ydelta, range = self.range, density=self.density, rwidth = self.rwidth, bins=self.bins)
        self.FVec = np.append(xcounts, ycounts, axis = 0)
        #self.FVec.append(xcounts)
        #self.FVec.append(ycounts)
        return self.FVec
        
    def loadhist(location):
        infile = open(location + '/histogram.pkl','rb')
        loaded = pickle.load(infile)
        return loaded

    def loadfvec(location):
        infile = open(location + '/fvec.pkl','rb')
        loaded = pickle.load(infile)
        return loaded
        
    def savehist(self, location):
        print('saving histogram')
        savefile = open(location + '/histogram.pkl', 'wb')
        savefile2 = open(location + '/fvec.pkl', 'wb')
        pickle.dump(self, savefile)
        pickle.dump(self.FVec, savefile2)
        savefile.close()
        savefile2.close()

        

HISTOGRAM = SURF_Histogram()


################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def SURF_FindMatches(kp1, kp2, desc1, desc2, n, i0, i1):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    matches = bf.match(desc1,desc2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    #matches[len(matches)-1].distance --> will give the greatest displacement
    newmatches = []
    
    # Error correction --> only consider the points with a correct match
    for match in matches:
        if abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1]) < 50:
            newmatches.append(match)
    
    
    #img3 = cv2.drawMatches(i0,kp,i1,kp2,newmatches[:], None, (0,255,0), flags=2)
    #img4 = cv2.drawMatches(i0,kp,i1,kp2,matches[:], None, (0,255,0), flags=2)
    #print('m', len(matches))
    #plt.imshow(img3),plt.show()
    
    #im = Image.fromarray(img3)
    #im.save('linesdrawn.jpeg')
    
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
    for i in kp1:
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
        print('kp1[i].pt[0]', kp1[i].pt[0])
        a = kp1[i].pt[0]/(xmax1 - xmin1 + 1)
        print('a1', a)
        print('kp1[i].pt[1]', kp1[i].pt[1])
        b = kp1[i].pt[1]/(ymax1 - ymin1 + 1)
        print('b1', b)
        pic1points.append((a,b))
    
    #print('pic2idxs',pic2idxs)
    #print('kp2', kp2)
    #print('len(kp2)', len(kp2))
    #print(kp2[i].pt[0])
    #print(kp2[i].pt[1])
    try:
        for i in pic2idxs:
            #print('i', i)
            print('kp2[i].pt[0]', kp2[i].pt[0])
            a = kp2[i].pt[0]/(xmax2 - xmin2 + 1)
            print('a2', a)
            print('kp2[i].pt[1]', kp2[i].pt[1])
            b = kp2[i].pt[1]/(ymax2 - ymin2 + 1)
            print('b2', b)
            pic2points.append((a,b))
    except:
        img3 = cv2.drawMatches(i0,kp1,i1,kp2,newmatches[:], None, (0,255,0), flags=2)
        img4 = cv2.drawMatches(i0,kp1,i1,kp2,matches[:], None, (0,255,0), flags=2)
        print('m', len(newmatches))
        for match in newmatches:
            print('query', match.queryIdx)
            print('train', match.trainIdx)
        plt.imshow(img3),plt.show()
        plt.imshow(img4),plt.show()
        while True:
            cv2.imshow('i0', i0)
            cv2.imshow('i1', i1)
            cv2.imshow('img3', img3)
            cv2.imshow('img4', img4)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"): # Exit condition
                cv2.destroyAllWindows()
                break

    
    # FOR THE DELTA, SHOULD WE TAKE THE ABSOLUTE VALUE??
    for i in range(len(newmatches)):
        if -1 < (pic1points[i][0] - pic2points[i][0]) < 1:
            xdelta.append(pic1points[i][0] - pic2points[i][0])
        if -1 < (pic1points[i][1] - pic2points[i][1]) < 1:
            ydelta.append(pic1points[i][1] - pic2points[i][1])

        
    return xdelta, ydelta

##------------------------------------------------------------------------------- (DONE)
    







################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def SURF_AddToHistograms(xdelta, ydelta):
    
    global HISTOGRAM
    
    HISTOGRAM.addx(xdelta)
    HISTOGRAM.addy(ydelta)    
##------------------------------------------------------------------------------- (DONE)



## Getting Directories
##=============================================================================
LABELS = []
LABELS_W = []
PATH = os.getcwd().replace('\\', '/')
print('PATH:\t\t', PATH)

AD_PATH = PATH + '/AD'
print('AD_PATH:\t', AD_PATH)

for filename in os.listdir(PATH_SURF):
    #print('--------------')
    if (re.match('.*_W', filename)):
        LABELS_W.append(filename)
    else:
        LABELS.append(filename)
    
##print(LABELS_W)
##print(LABELS)
#
##***
## By now, we have all the labels split into the arrays based on depthwhite or gray frames
##***
###-----------------------------------------------------------------------------
#
##print('---------------------')
#
#
#
#
## Training on Depthwhite frames
##=============================================================================

#for label in LABELS:                                          # open each depthwhite recorded person
#    walks = []                                              # array to store how many times they have walked
#    label_path = PATH_SURF + '/' + label                          # path to that person
#    for walk in os.listdir(label_path):
#        print('SURF for ' + str(walk) + ' for ' + str(label))
#        walks.append(walk)                                  # get each walk
#        walk_path = label_path + '/' + walk                 # path to that walk
#        
#        NUMPYLIST = []
#        for frame in os.listdir(walk_path + '/ORIGINAL'):
#            print(frame)
#            openednumpyImage = cv2.imread(walk_path + '/ORIGINAL/' + str(frame))
#            NUMPYLIST.append(openednumpyImage)
#        
#        HISTOGRAM.resetx()
#        HISTOGRAM.resety()
#        HISTOGRAM.resetfvec()
#        
#        surf = cv2.ORB_create(nfeatures = 100)
#            
#        for i in range(len(NUMPYLIST) - 1):
#            print(i, 'called')
#            
#            kp1, desc1 = surf.detectAndCompute(NUMPYLIST[i], None)
#            kp2, desc2 = surf.detectAndCompute(NUMPYLIST[i+1], None)
#            
#            i0 = cv2.drawKeypoints(NUMPYLIST[i], kp1, None, (255,0,0), 4)
#            i1 = cv2.drawKeypoints(NUMPYLIST[i+1], kp2, None, (255,0,0), 4)
#            
#            xdelta, ydelta = SURF_FindMatches(kp1, kp2, desc1, desc2, i, i0, i1)
#            
#            SURF_AddToHistograms(xdelta, ydelta)
#            
#            print('Pair finished-----------------------')
#            
#        fvec = HISTOGRAM.getfeaturevec()
#        print(fvec)
#        print('Walk finished-----------------------')
#        HISTOGRAM.savehist(label_path + '/' + walk)
##-----------------------------------------------------------------------------

        
def checkPlain(im):
    count = 0
    for w in range(width):
        for h in range(height):
            if int(im[h,w]) > 0:
                count += 1
                if count >= 5000:
                    return False
    return True
        



first = True
second = True
traininglabels = []
mainvec = []
firstvec = None
secondvec = None
def SURF_fvecs():

    for i in LABELS:                                          # open each depthwhite recorded person
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_SURF + '/' + i                          # path to that person
        
        
        
        for walk in os.listdir(label_path):
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            
            FVec = SURF_Histogram.loadfvec(walk_path)
            
            FVec = np.array(FVec)
            FVec = FVec.reshape(1,-1)

            
            global first, second, firstvec, secondvec, mainvec
            if first and second:
                firstvec = FVec
                first = False
            elif second:
                secondvec = FVec
                mainvec = np.append(firstvec, secondvec, axis = 0)
                second = False
            else:
                mainvec = np.append(mainvec, FVec, axis = 0)
            traininglabels.append(i)
#    clf = svm.SVC(kernel='linear', C = 1)
#    clf.fit(mainvec, traininglabels)
#    savefile = open(location + '/histogram.pkl', 'wb')
#    pickle.dump(self, savefile)
#    savefile.close()
SURF_fvecs()

tFinal = time()
print("Linear SVM training time = ", round(tFinal-tInit, 5), "s")

#

#clf = svm.SVC(kernel='linear', C = 1)
#clf.fit(mainvec, traininglabels)
#
#
##y_pred = clf.predict(featurevecsw4)
##print(y_pred)
#
#
#testvec = SURF_Histogram.loadfvec(r'C:\Users\Asif Towheed\Desktop\Trimmed\SURF\WALK2')
#testvec = np.array(testvec)
#testvec = testvec.reshape(1,-1)
#
#y_pred = clf.predict(testvec)
#print(y_pred)


X_train, X_test, y_train, y_test = train_test_split(mainvec, traininglabels, test_size = 0.10)  

svclassifier = svm.SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)  

cm = confusion_matrix(y_test,y_pred)
print(cm)  
cr = classification_report(y_test,y_pred)
print(cr)
score = accuracy_score(y_test, y_pred)
print(score)












