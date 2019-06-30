# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:04:12 2019

@author: Asif Towheed
"""
import datetime
import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from sklearn import svm
import math
import re
import pickle
from time import time

LABELS = []
LABELS_ALL = []
TRAINING_STATUS = False

PATH = os.getcwd().replace('\\','/')
PATH_AD = PATH + '/AD'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALORIGINALPATH = ""
FINALDIFFPATH = ""
INDEX = 0

width, height = (640, 480)

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
    
    print('AD_Begin()')
    
    ## Getting Directories
    ##=============================================================================
    global LABELS, LABELS_ALL
    PATH = os.getcwd().replace('\\', '/')
    print('PATH:\t\t', PATH)
    
    AD_PATH = PATH + '/AD'
    print('AD_PATH:\t', AD_PATH)
    
    for filename in os.listdir(AD_PATH):
        #print('--------------')
#        if (re.match('.*_W', filename)):
#            LABELS_W.append(filename)
        if re.match('.*_untrained', filename):
            LABELS.append(filename)
        elif not re.match('trained-model', filename):
            LABELS_ALL.append(filename)
            
            
    # Training on frames
    #=============================================================================
    
    for i in LABELS:                                          # open each recorded person
        global SUBJECTNAME, KILLTHREAD
        SUBJECTNAME = i
        if KILLTHREAD:
            return
        AD_GenerateDiffs(SUBJECTNAME)
        ts = time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + '\n'
        file = open(PATH + '/history.txt', 'r+')
        contents_as_string = file.read()
        newtext = SUBJECTNAME.split('_untrained')[0] + ' was trained for AD: ' + timestamp + contents_as_string
        file2 = open(PATH + '/history.txt', 'w')
        file2.write(newtext)
        file.close()
        file2.close()

    #-----------------------------------------------------------------------------
##-------------------------------------------------------------------------------

def getStatus2(timeinit):
    tfinal = time()
    global TRAINING_STATUS
    if not TRAINING_STATUS:
        return str(tfinal-timeinit) + 's; Processing walk ' + str(INDEX) + ' for ' + str(SUBJECTNAME) + '...'
    else:
        return 'Training the classifier...'






################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def AD_FindDiff(im1, im2, n):
    global ACC_DIFF_IMG, STATUSARR
    IND_DIFF_IMG = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    sumofdiff = 0
    
    f = np.absolute(np.subtract(im1, im2))
    f[f < 30] = 0
    IND_DIFF_IMG = f
    
    sumofdiff = np.sum(f)
    
#    for w in range(width):
#        for h in range(height):
#            absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
#            #print("PRINTING", int(im1[h,w]))
#            if absdiff < 30:
#                absdiff = 0
#            sumofdiff += absdiff
#            IND_DIFF_IMG[h,w] = absdiff
    
    newthresh = sumofdiff/(width*height)
    
    IND_DIFF_IMG[IND_DIFF_IMG < newthresh] = 0
    ACC_DIFF_IMG = np.where(IND_DIFF_IMG > newthresh, ACC_DIFF_IMG + 10, ACC_DIFF_IMG)

    
#    for w in range(width):
#        for h in range(height):
#            if IND_DIFF_IMG[h,w] < newthresh:
#                IND_DIFF_IMG[h,w] = 0
#            else:
#                ACC_DIFF_IMG[h,w] += 10
    #print("SAVING")
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
    
    print('AD_GD()')
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
            if num%3 == 0:
                openedImage = Image.open(walk_path + '/ORIGINAL/' + frame).convert("L")
                #print(walk_path + '/ORIGINAL/' + frame)
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
    print('SUBJECTPATH', SUBJECTPATH)
    os.rename(SUBJECTPATH, SUBJECTPATH.split('_untrained')[0])
    LABELS_ALL.append(subjectname.split('_untrained')[0])
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

loadfile = None

def AD_fvecs():
    global TRAINING_STATUS, LABELS_ALL
    TRAINING_STATUS = True
    print('AD_fvecs()')
    first = True
    second = True
    traininglabels = []
    mainvec = None
    firstvec = None
    secondvec = None
    print(LABELS_ALL)
    for i in LABELS_ALL:                                        # open each depthwhite recorded person
#        print('enter')
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_AD + '/' + i                          # path to that person
        
        
        
        for walk in os.listdir(label_path):
#            -- code for cross testing --> TRAIN ON JEANS
#            if os.path.exists('C:/Users/Asif Towheed/Desktop/Trimmed/K_A/' + i + '/' + walk):
#                break
#            -- code for cross testing --> TRAIN ON K/A
            if not os.path.exists('C:/Users/Asif Towheed/Desktop/Trimmed/K_A/' + i + '/' + walk) and not i in ['ABDALLAH','HIND','HANIA','SOMAR','REZA']:
                continue
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            
            # Getting the DCTs
            adtemp = np.array(Image.open(walk_path + '/AD.jpeg').convert("L"))
#            print(adtemp)
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
        print(i, walks)
    clf = svm.SVC(kernel='linear', C = 1)
    clf.fit(mainvec, traininglabels)
    savefile = open(PATH_AD + '/trained-model.pkl', 'wb')
    pickle.dump(clf, savefile)
    savefile.close()
    somefile = open(PATH_AD + '/trained-model.pkl', 'rb')
    global loadfile
    loadfile = pickle.load(somefile)
    #loadfile.close()
    TRAINING_STATUS = False

#AD_fvecs()




    
    
    
    
#AD_Begin('Hello')