# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:04:12 2019

@author: Asif Towheed
"""

import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from sklearn import svm
import math
import re
import pickle

LABELS = []
LABELS_W = []

PATH = os.getcwd().replace('\\','/')
PATH_AD = PATH + '/AD'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALORIGINALPATH = ""
FINALDIFFPATH = ""
INDEX = 0

cdef int width = 640
cdef int height = 480


IMAGELIST = []
NUMPYLIST = []
ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
FEATUREVEC = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []
PAUSE = False

print(PATH)
print(PATH_AD)

################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def AD_Begin():
    
    ## Getting Directories
    ##=============================================================================
    global LABELS, LABELS_W
    PATH = os.getcwd().replace('\\', '/')
    print('PATH:\t\t', PATH)
    
    AD_PATH = PATH + '/AD'
    print('AD_PATH:\t', AD_PATH)
    
    for filename in os.listdir(AD_PATH):
        #print('--------------')
        if (re.match('.*_W', filename)):
            LABELS_W.append(filename)
        elif not re.match('trained-model', filename):
            LABELS.append(filename)
            
            
    # Training on frames
    #=============================================================================
    
    for i in LABELS:                                          # open each recorded person
        global SUBJECTNAME, KILLTHREAD
        SUBJECTNAME = i
        if KILLTHREAD:
            return
        AD_GenerateDiffs(SUBJECTNAME)
    #-----------------------------------------------------------------------------
##-------------------------------------------------------------------------------


def getStatus2():
    return 'Processing walk ' + str(INDEX) + ' for ' + str(SUBJECTNAME) + '...'





################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def AD_FindDiff(im1, im2, n):
    global ACC_DIFF_IMG, STATUSARR
    IND_DIFF_IMG = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    cdef int sumofdiff = 0
    
    cdef int w = 0
    cdef int h = 0
    cdef float newthresh = 0
    
    for w in range(width):
        for h in range(height):
            absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
            #print("PRINTING", int(im1[h,w]))
            if absdiff < 30:
                absdiff = 0
            sumofdiff += absdiff
            IND_DIFF_IMG[h,w] = absdiff
    
    newthresh = sumofdiff/(width*height)
    
    w = 0
    h = 0

    for w in range(width):
        for h in range(height):
            if IND_DIFF_IMG[h,w] < newthresh:
                IND_DIFF_IMG[h,w] = 0
            else:
                ACC_DIFF_IMG[h,w] += 10
    print("SAVING22")
    newimage = Image.fromarray(IND_DIFF_IMG)
    newimage.save(FINALDIFFPATH + "/IND-IMAGE-{}.jpeg".format(n))
    STATUSARR.append(n)

##-------------------------------------------------------------------------------




def AD_GetWalkNumber():
    return int(INDEX)

#def AD_FinishedArrs():
    



################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def AD_GenerateDiffs(subjectname):
    
    walks = []                                              # array to store how many times they have walked
    global IMAGELIST, STATUSARR, INDEX, FINALORIGINALPATH
    try:
        global FINALDIFFPATH
        SUBJECTPATH = PATH_AD + '/' + subjectname
        
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')        

    for walk in os.listdir(SUBJECTPATH):
        global PAUSE, STATUSBOOL


        walks.append(walk)                                  # get each walk
        walk_path = SUBJECTPATH + '/' + walk                 # path to that walk
        INDEX = walk[-1]
        FINALDIFFPATH = walk_path + "/DIFFERENCES"
        FINALORIGINALPATH = walk_path + "/ORIGINAL"
        
        
        
        STATUSBOOL = False
        STATUSARR = []        
        IMAGELIST = []
        NUMPYLIST = []
        num = 0
        for frame in os.listdir(walk_path + '/ORIGINAL'):
            if num%5 == 0:
                openedImage = Image.open(walk_path + '/ORIGINAL/' + frame).convert("L")
                print(walk_path + '/ORIGINAL/' + frame)
                IMAGELIST.append(openedImage)
                NUMPYLIST.append(np.array(openedImage))
            num += 1
        
        if not os.path.exists(FINALDIFFPATH):
            os.mkdir(FINALDIFFPATH)

        STATUSBOOL = True
        PAUSE = False

        k = 0
        for j in range(len(NUMPYLIST) - 1):
            k += 1
            AD_FindDiff(NUMPYLIST[j], NUMPYLIST[j+1], k)
            global KILLTHREAD
            if KILLTHREAD:
                print('Diff1 stopped')                
                return 

        PAUSE = True
        
        global ACC_DIFF_IMG
        newimage = Image.fromarray(ACC_DIFF_IMG)
        newimage.save(walk_path + "/AD.jpeg")
        ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
        
    print('DIFFERENCES STOPPED')
    MID_AD_RESET()

##-------------------------------------------------------------------------------

    
    



    

################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def AD_GetStatus():
    global IMAGELIST, NUMPYLIST, STATUSBOOL, FINALDIFFPATH, STATUSARR, PAUSE
    
    if not STATUSBOOL:
        return 0
    elif PAUSE:
        return 20
    else:
        #print('len(STATUSARR)', len(STATUSARR))
        #print('(len(IMAGELIST)-1)', (len(IMAGELIST)-1))
        #print('len(STATUSARR)/(len(IMAGELIST)-1) * 20',len(STATUSARR)/(len(IMAGELIST)-1) * 20)
        return int(math.ceil(len(STATUSARR)/(len(IMAGELIST)-1) * 20))
##-------------------------------------------------------------------------------

    
    



    
    
################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def AD_GetDCT():
    global ACC_DIFF_IMG, FEATUREVEC

    DCT_Image = dct( dct( ACC_DIFF_IMG, axis=0), axis=1)

    for i in range(10):
        for j in range(10):
            FEATUREVEC.append(DCT_Image[i][j])

##-------------------------------------------------------------------------------

    
    



    
    
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
def AD_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD
    PATH = os.getcwd().replace('\\','/')
    PATH_AD = PATH + '/AD'
    SUBJECTNAME = ""
    SUBJECTPATH = ""
    FINALORIGINALPATH = ""
    FINALDIFFPATH = ""
    INDEX = 0
        
    IND_DIFF_IMG = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
    FEATUREVEC = []
    KILLTHREAD = False
    print('RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------
################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def MID_AD_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD, STATUSBOOL, STATUSARR

    FEATUREVEC = []
    KILLTHREAD = False
    STATUSBOOL = False
    STATUSARR = []
    INDEX = 0

    print('MID-RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------



def AD_fvecs():
    first = True
    second = True
    traininglabels = []
    mainvec = None
    firstvec = None
    secondvec = None
    for i in LABELS:                                          # open each depthwhite recorded person
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_AD + '/' + i                          # path to that person
        
        
        
        for walk in os.listdir(label_path):
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            
            # Getting the DCTs
            adtemp = np.array(Image.open(walk_path + '/AD.jpeg').convert("L"))
            #print(adtemp)
            dcttemp = dct( dct( adtemp, axis=0), axis=1)
            
            # Getting the FVecs
            featurevec = []
            for m in range(10):
                for n in range(10):
                    featurevec.append(dcttemp[m][n])
            
            FVec = np.array(featurevec)
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
    savefile = open(PATH_AD + '/trained-model.pkl', 'wb')
    pickle.dump(clf, savefile)
    savefile.close()






    
    
    
    
#AD_Begin('Hello')