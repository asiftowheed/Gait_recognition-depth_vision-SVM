# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:11:57 2019

@author: Asif Towheed
"""

import datetime
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
import math



width, height = (640, 480)

LABELS = []
LABELS_ALL = []
TRAINING_STATUS = False

PATH = os.getcwd().replace('\\','/')
PATH_SURF = PATH + '/SURF'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALPATH = ""
INDEX = 0

width, height = (640, 480)

XDELTA = []
YDELTA = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []
NUMPYLIST = []

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
#    print('xmax1', xmax1)
#    print('ymax1', ymax1)
#    print('xmax2', xmax2)
#    print('ymax2', ymax2)
#    print('xmin1', xmin1)
#    print('ymin1', ymin1)
#    print('xmin2', xmin2)
#    print('ymin2', ymin2)
    
    # NORMALIZATION
    #print('pic1idxs',pic1idxs)
    for i in pic1idxs:
#        print('kp1[i].pt[0]', kp1[i].pt[0])
        a = kp1[i].pt[0]/(xmax1 - xmin1 + 1)
#        print('a1', a)
#        print('kp1[i].pt[1]', kp1[i].pt[1])
        b = kp1[i].pt[1]/(ymax1 - ymin1 + 1)
#        print('b1', b)
        pic1points.append((a,b))
    
    #print('pic2idxs',pic2idxs)
    #print('kp2', kp2)
    #print('len(kp2)', len(kp2))
    #print(kp2[i].pt[0])
    #print(kp2[i].pt[1])
    try:
        for i in pic2idxs:
            #print('i', i)
#            print('kp2[i].pt[0]', kp2[i].pt[0])
            a = kp2[i].pt[0]/(xmax2 - xmin2 + 1)
#            print('a2', a)
#            print('kp2[i].pt[1]', kp2[i].pt[1])
            b = kp2[i].pt[1]/(ymax2 - ymin2 + 1)
#            print('b2', b)
            pic2points.append((a,b))
    except:
        img3 = cv2.drawMatches(i0,kp1,i1,kp2,newmatches[:], None, (0,255,0), flags=2)
        img4 = cv2.drawMatches(i0,kp1,i1,kp2,matches[:], None, (0,255,0), flags=2)
#        print('m', len(newmatches))
#        for match in newmatches:
#            print('query', match.queryIdx)
#            print('train', match.trainIdx)
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



################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def SURF_Begin():
    
    ## Getting Directories
    ##=============================================================================
    global LABELS, LABELS_ALL

    PATH = os.getcwd().replace('\\', '/')
    print('PATH:\t\t', PATH)
    
    SURF_PATH = PATH + '/SURF'
    print('SURF_PATH:\t', SURF_PATH)
    
    for filename in os.listdir(PATH_SURF):

#        if not re.match('trained-model', filename):
#            LABELS.append(filename)

        if re.match('.*_untrained', filename):
            LABELS.append(filename)
        elif not re.match('trained-model', filename):
            LABELS_ALL.append(filename)
            
            
    # Training on frames
    #=============================================================================
    
    for i in LABELS:                                          # open each recorded person
        global SUBJECTNAME, KILLTHREAD
        SUBJECTNAME = i
        print('now for',SUBJECTNAME,'killthread',KILLTHREAD)
        if KILLTHREAD:
            print('Entered1')
            return
        SURF_GenerateHistograms(SUBJECTNAME)
        ts = time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + '\n'
        file = open(PATH + '/history.txt', 'r+')
        contents_as_string = file.read()
        newtext = SUBJECTNAME.split('_untrained')[0] + ' was trained for SURF: ' + timestamp + contents_as_string
        file2 = open(PATH + '/history.txt', 'w')
        file2.write(newtext)
        file.close()
        file2.close()
        #-----------------------------------------------------------------------------
##-------------------------------------------------------------------------------
    
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

def SURF_GenerateHistograms(SUBJECTNAME):                                      # open each depthwhite recorded person
    walks = []                                                      # array to store how many times they have walked
    SUBJECTPATH = PATH_SURF + '/' + SUBJECTNAME                            # path to that person
    for walk in os.listdir(SUBJECTPATH):
        global INDEX, STATUSBOOL, STATUSARR, NUMPYLIST
        print('SURF for ' + str(walk) + ' for ' + str(SUBJECTNAME))
        walks.append(walk)                                  # get each walk
        walk_path = SUBJECTPATH + '/' + walk                 # path to that walk
        
        INDEX = walk[-1]

        STATUSBOOL = False
        STATUSARR = []

        
        NUMPYLIST = []
        
        global FINALDIFFPATH, FINALORIGINALPATH, PADDEDPATH, PREPATH, KILLTHREAD
        FINALORIGINALPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/ORIGINAL"        
            
        for frame in os.listdir(FINALORIGINALPATH):
            #print(frame)
            openednumpyImage = cv2.imread(FINALORIGINALPATH + '/' + str(frame))
            NUMPYLIST.append(openednumpyImage)
        
        HISTOGRAM.resetx()
        HISTOGRAM.resety()
        HISTOGRAM.resetfvec()
        
        surf = cv2.ORB_create(nfeatures = 100)
        
        STATUSBOOL = True
            
        for i in range(len(NUMPYLIST) - 1):
            if KILLTHREAD:
                return
            #print(i, 'called')
            
            kp1, desc1 = surf.detectAndCompute(NUMPYLIST[i], None)
            kp2, desc2 = surf.detectAndCompute(NUMPYLIST[i+1], None)
            
            i0 = cv2.drawKeypoints(NUMPYLIST[i], kp1, None, (255,0,0), 4)
            i1 = cv2.drawKeypoints(NUMPYLIST[i+1], kp2, None, (255,0,0), 4)
            
            xdelta, ydelta = SURF_FindMatches(kp1, kp2, desc1, desc2, i, i0, i1)
            
            SURF_AddToHistograms(xdelta, ydelta)
            
            #print('Pair finished-----------------------')
            STATUSARR.append(1)
            
        fvec = HISTOGRAM.getfeaturevec()
        #print(fvec)
        #print('Walk finished-----------------------')
        HISTOGRAM.savehist(walk_path)

    print('SUBJECTPATH', SUBJECTPATH)
#    os.rename(SUBJECTPATH, SUBJECTPATH.split('_untrained')[0])
    MID_SURF_RESET()
##-----------------------------------------------------------------------------



################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def SURF_GetStatus():
    global IMAGELIST, NUMPYLIST, STATUSBOOL, FINALDIFFPATH, STATUSARR, PAUSE
    
    if not STATUSBOOL:
        return 0
#    elif PAUSE:
#        return 20
    else:
        #print('len(STATUSARR)', len(STATUSARR))
        #print('(len(IMAGELIST)-1)', (len(IMAGELIST)-1))
        #print('len(STATUSARR)/(len(IMAGELIST)-1) * 20',len(STATUSARR)/(len(IMAGELIST)-1) * 20)
        return int(math.ceil(len(STATUSARR)/(len(NUMPYLIST)-1) * 20))
##-------------------------------------------------------------------------------



def SURF_GetWalkNumber():
    return int(INDEX)

def getStatus2(timeinit):
    tfinal = time()
    global TRAINING_STATUS
    if not TRAINING_STATUS:
        return str(tfinal-timeinit) + 's; Processing walk ' + str(INDEX) + ' for ' + str(SUBJECTNAME) + '...'
    else:
        return 'Training the classifier...'




################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def TerminateDifferences():
    global KILLTHREAD
    KILLTHREAD = True
##-------------------------------------------------------------------------------

################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def SURF_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD
    PATH = os.getcwd().replace('\\','/')
    PATH_AD = PATH + '/AD'
    SUBJECTNAME = ""
    SUBJECTPATH = ""
    FINALORIGINALPATH = ""
    FINALDIFFPATH = ""
    INDEX = None
        
    IND_DIFF_IMG = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
    FEATUREVEC = []
    KILLTHREAD = False
    print('RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------
################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def MID_SURF_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD, STATUSBOOL, STATUSARR

    FEATUREVEC = []
    KILLTHREAD = False
    STATUSBOOL = False
    STATUSARR = []
    INDEX = 0

    print('MID-RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------





        
def checkPlain(im):
    count = 0
    for w in range(width):
        for h in range(height):
            if int(im[h,w]) > 0:
                count += 1
                if count >= 5000:
                    return False
    return True
        


def SURF_train():
    global TRAINING_STATUS
    TRAINING_STATUS = True
    first = True
    second = True
    traininglabels = []
    mainvec = []
    firstvec = None
    secondvec = None
    for i in LABELS:                                          # open each depthwhite recorded person
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_SURF + '/' + i                          # path to that person
        
        
        
        for walk in os.listdir(label_path):
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            
            FVec = SURF_Histogram.loadfvec(walk_path)
            
            FVec = np.array(FVec)
            FVec = FVec.reshape(1,-1)

            
            #global first, second, firstvec, secondvec, mainvec
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
    clf = svm.SVC(kernel='linear', C = 1)
    clf.fit(mainvec, traininglabels)
    savefile = open(PATH_SURF + '/trained-model.pkl', 'wb')
    pickle.dump(clf, savefile)
    savefile.close()
    TRAINING_STATUS = False



