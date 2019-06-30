# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:51:27 2019

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

PATH = os.getcwd().replace('\\','/')
PATH_GEI = PATH + '/GEI'
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
PAUSE = False
TRAINING_STATUS = False

tInit = time()


PADDEDPATH = None
PREPATH = None

width, height = (640, 480)
ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)


################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def GEI_Begin():
    
    ## Getting Directories
    ##=============================================================================
    global LABELS, LABELS_ALL

    PATH = os.getcwd().replace('\\', '/')
#    print('PATH:\t\t', PATH)
    
    GEI_PATH = PATH + '/GEI'
#    print('GEI_PATH:\t', GEI_PATH)
    
    for filename in os.listdir(PATH_GEI):
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
        GEI_Preprocess(SUBJECTNAME)
        ts = time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + '\n'
        file = open(PATH + '/history.txt', 'r+')
        contents_as_string = file.read()
        newtext = SUBJECTNAME.split('_untrained')[0] + ' was trained for GEI: ' + timestamp + contents_as_string
        file2 = open(PATH + '/history.txt', 'w')
        file2.write(newtext)
        file.close()
        file2.close()
    #-----------------------------------------------------------------------------
##-------------------------------------------------------------------------------
    
    

################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def GEI_FindDiff(im1, im2, n):
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
    ACC_DIFF_IMG = np.where(IND_DIFF_IMG >= newthresh, ACC_DIFF_IMG + 10, ACC_DIFF_IMG)

    
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
##------------------------------------------------------------------------------- (DONE)
    

################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def GEI_GenerateGEI(SUBJECTNAME):

#    try:
#        global FINALDIFFPATH, FINALORIGINALPATH, PADDEDPATH, PREPATH
#        FINALORIGINALPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/ORIGINAL"
#        FINALDIFFPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/DIFFERENCES"
#        PADDEDPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/PADDED"
#        PREPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/PRE-FOR-GEI"
#        os.mkdir(FINALDIFFPATH)
#    except OSError:
#        print('Error in creation of directory')
#    else:
#        print('Successfully created directory')    
#
#    print('---------EXTRACTING IMAGES INTO AN ARRAY--------------------------------------------')
    
    global IMAGELIST, NUMPYLIST
    IMAGELIST = []
    NUMPYLIST = []

    sumw = sumh = maxw = maxh = 0
    count = 0
    for filename in os.listdir(PREPATH):
        #print(filename)
#        openedImage = Image.open(FINALORIGINALPATH + '/' + str(filename)).convert("L")
#        IMAGELIST.append(openedImage)
#        NUMPYLIST.append(np.array(openedImage))
        #                 for filepath in glob.iglob(PATH + '/*.jpeg'):
        #path = PATH + '/*.jpeg'
#        print(filepath)
#        print(path)

        if re.match('Pre-processed_.*',filename):
            #print(filename)
            ori = cv2.imread(PREPATH + '/' + filename)
            height, width, depth = ori.shape
            sumw += width
            sumh += height
            if width > maxw:
                maxw = width
            if height > maxh:
                maxh = height
            #print(width, height)
            count += 1

    #avgw = sumw/count
    #avgh = sumh/count
    #print('count', count)
    result = None
    checker = True

    npimages = []


    for filename in os.listdir(PREPATH):
        if KILLTHREAD:
            return
        if re.match('Pre-processed_.*',filename):
            ori = cv2.imread(PREPATH + '/' + filename)
            height, width, depth = ori.shape
            #print(width, height)
            #print('---',filename,'---')
#            if width < 150 or width > 300:
#                dummy = 0
#                #print('removing', filename)
#                #os.unlink(filename)
#            else:
            if not width < 150 and not width > 300:
                to_be_added_w = (640-width)
                to_be_added_h = (480-height)                
                #print('adding', filename)
                #newimg = cv2.resize(ori,(int(maxw),int(maxh)))
                #print('maxw:', maxw)
                #print('width:', width)
                #print('to_be_added_w:', to_be_added_w)
                padval = math.floor(to_be_added_w/2)
                #print("PADVAL ", padval)
                if (to_be_added_w%2) == 0:
                    #print("ENTERED!!!!!!!!!!!!!!")
                    newimg = np.pad(ori, ((to_be_added_h,0),(padval,padval), (0,0)), mode="constant")
                else:
                    #print("22222ENTERED!!!!!!!!!!!!!!")
                    newimg = np.pad(ori, ((to_be_added_h,0),(padval,padval+1), (0,0)), mode="constant")
                #newimg = cv2.resize(newimg,(int(maxw),int(maxh)))
                if checker is True:
                    #print(checker)
                    #print("ENTERED!!!!!!!!!!!!")
                    result = Image.fromarray(newimg)
                    checker = False
                    #print(checker)
#                pilimg = Image.fromarray(newimg)
#                pilimg.save(filename)
                npimages.append(newimg)
            STATUSARR.append(2)

    #print('npimages:', npimages)
    #print('lenth of npimages:', len(npimages))
    for i in range(len(npimages)-1):
        #newimg = finddiff(npimages[i], npimages[i+1])
        #h1, w1 = np.array(result).size
        #h2, w2 = newimg.size
        #print("h1",h1,"w1",w1," h2",h2,"w2",w2)
        #print(result == None)
        if KILLTHREAD:
            return
        #print('-----------------------------')
        pilimg = Image.fromarray(np.array(result))
        pilimg.save(PADDEDPATH + '/result.jpeg')
        pilimg = Image.fromarray(np.array(npimages[i]))
        pilimg.save(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        newimg = cv2.imread(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        result = cv2.imread(PADDEDPATH + '/result.jpeg')
        
        #print(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        result = cv2.addWeighted(result, 0.90, newimg, 0.10, 0)    
       
    global STATUSBOOL
    STATUSBOOL = False

#    for i in range(len(NUMPYLIST) - 1):
#        print(i, 'called')
#        GEI_FindDiff(NUMPYLIST[i], NUMPYLIST[i+1], i)

##-------------------------------------------------------------------------------


################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def GEI_Preprocess(SUBJECTNAME):

    walks = []                                              # array to store how many times they have walked
    global IMAGELIST, STATUSARR, INDEX, FINALORIGINALPATH
    try:
        global FINALDIFFPATH
        SUBJECTPATH = PATH_GEI + '/' + SUBJECTNAME
        
    except OSError:
        print('Error in creation of directory')
    else:
        DUMMY = 0
#        print('Successfully created directory')        

    for walk in os.listdir(SUBJECTPATH):
        global PAUSE, STATUSBOOL


        walks.append(walk)                                  # get each walk
        walk_path = SUBJECTPATH + '/' + walk                 # path to that walk
        INDEX = walk[-1]
        
        STATUSBOOL = False
        STATUSARR = []
        IMAGELIST = []
        NUMPYLIST = []
        
        try:
            #SUBJECTPATH
            global FINALDIFFPATH, FINALORIGINALPATH, PADDEDPATH, PREPATH
            FINALORIGINALPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/ORIGINAL"
            FINALDIFFPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/DIFFERENCES"
            PADDEDPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/PADDED"
            PREPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/PRE-FOR-GEI"
            if not os.path.exists(FINALDIFFPATH):
                os.mkdir(FINALDIFFPATH)
            if not os.path.exists(PADDEDPATH):
                os.mkdir(PADDEDPATH)
            if not os.path.exists(PREPATH):
                os.mkdir(PREPATH)
        
        except OSError:
            print('Error in creation of directory')
        else:
            DUMMY = 0
#            print('Successfully created directory')    
    
#        print('---------EXTRACTING IMAGES INTO AN ARRAY--------------------------------------------')
    
        STATUSBOOL = True
    
        
        #----------------------------------------------------------------------
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

#        PAUSE = False

        k = 0
        for j in range(len(NUMPYLIST) - 1):
            k += 1
            GEI_FindDiff(NUMPYLIST[j], NUMPYLIST[j+1], k)
            global KILLTHREAD
            if KILLTHREAD:
                print('Diff1 stopped')                
                return 
        #----------------------------------------------------------------------

        
        
        
#        print('GEI_Preprocess foP:', FINALORIGINALPATH)
#        num = 0
        for filename in os.listdir(FINALDIFFPATH):
#            if num%3 == 0:
#            print('FILENAMING', filename)
            if KILLTHREAD:
                return
            imagearr2 = np.array(Image.open(FINALDIFFPATH + '/' + str(filename)).convert("L"))
            lowest = rightmost = 1
            topmost = 480
            leftmost = 640

#            imagearr2[imagearr2 != 0] = 255
            
            x, y = np.nonzero(imagearr2)
#                print(x)
            
            topmost = min(x)
            lowest = max(x)
            leftmost = min(y)
            rightmost = max(y)
            detected = True
            
#                print('detected', detected)
#                print('leftmost != rightmost', leftmost != rightmost)
#                print('topmost != lowest', topmost != lowest)
            
            if detected is True and leftmost != rightmost and topmost != lowest:
#                    print('Enter')
                bbox = (leftmost, topmost, rightmost, lowest)
                #print(bbox)
                #person = imagearr[topmost:lowest-topmost, leftmost:rightmost-leftmost]
                pilimg = Image.fromarray(imagearr2).crop(bbox)
        #        print("saving")
    #            global l
    #            l = l + 1
        #        pilimg.show()
                pilimg.save(PREPATH + '/Pre-processed_' + filename)
#                print('saved preprocessed ' + filename + ' at ' + PREPATH)
            STATUSARR.append(1)
#            num += 1
                
        GEI_GenerateGEI(SUBJECTNAME)

    print('SUBJECTPATH', SUBJECTPATH)
    os.rename(SUBJECTPATH, SUBJECTPATH.split('_untrained')[0])
    LABELS_ALL.append(SUBJECTNAME.split('_untrained')[0])
    MID_GEI_RESET()
#    print('---------FOR FOLDERS --------------------------------------------')
#    for filename in os.listdir(PATH):
#        if not os.path.isfile(filename):
#            print(filename)
#        
#    print('---------FOR FILES --------------------------------------------')
#    for filename in os.listdir(PATH):
#        if os.path.isfile(filename):
#            print(filename)
##-------------------------------------------------------------------------------


################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def GEI_GetStatus():
    global IMAGELIST, NUMPYLIST, STATUSBOOL, FINALDIFFPATH, STATUSARR, PAUSE
    
    if not STATUSBOOL:
        return 0
#    elif PAUSE:
#        return 20
    else:
        #print('len(STATUSARR)', len(STATUSARR))
        #print('(len(IMAGELIST)-1)', (len(IMAGELIST)-1))
        #print(len(os.listdir(FINALORIGINALPATH)) * 2)
        #print(len(STATUSARR))
#        print("(len(os.listdir(FINALORIGINALPATH)) * 2) * 20",(len(os.listdir(FINALORIGINALPATH)) * 2) * 20)
        return int(math.ceil((len(STATUSARR)/(len(os.listdir(FINALORIGINALPATH)) * 2)) * 20))
#        return 20
##-------------------------------------------------------------------------------

def GEI_GetWalkNumber():
    return int(INDEX)

def getStatus2(timeinit):
    tfinal = time()
    global TRAINING_STATUS
    if not TRAINING_STATUS:
        return str(tfinal-timeinit) + 's; Processing walk ' + str(INDEX) + ' for ' + str(SUBJECTNAME) + '...'
    else:
        return 'Training the classifier...'


    
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
#    label_path = PATH_GEI + '/' + label                          # path to that person
#    for walk in os.listdir(label_path):
#        print('GEI for ' + str(walk) + ' for ' + str(label))
#        walks.append(walk)                                  # get each walk
#        walk_path = label_path + '/' + walk                 # path to that walk
#        
#        os.mkdir(walk_path + '/PADDED')
#        os.mkdir(walk_path + '/PRE-FOR-GEI')
#        
#        PADDEDPATH = walk_path + '/PADDED'
#        PREPATH = walk_path + '/PRE-FOR-GEI'
#        GEI_Preprocess(walk_path + '/ORIGINAL', PADDEDPATH, PREPATH)
#        GEI_GenerateGEI(walk_path + '/ORIGINAL', PADDEDPATH, PREPATH)
#            
#        print('Walk finished-----------------------')
##-----------------------------------------------------------------------------


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
def GEI_RESET():
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
def MID_GEI_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD, STATUSBOOL, STATUSARR

    FEATUREVEC = []
    KILLTHREAD = False
    STATUSBOOL = False
    STATUSARR = []
    INDEX = 0

#    print('MID-RESET CALLED', str(KILLTHREAD))
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
        


def GEI_fvecs():
    
    global TRAINING_STATUS
    TRAINING_STATUS = True
    first = True
    second = True
    traininglabels = []
    mainvec = []
    firstvec = None
    secondvec = None
    for i in LABELS_ALL:                                        # open each depthwhite recorded person
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_GEI + '/' + i                         # path to that person
        
        
        
        for walk in os.listdir(label_path):
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            
            PADDEDPATH = walk_path + '/PADDED'

            # Getting the DCTs
            resulttemp = np.array(Image.open(PADDEDPATH + '/result.jpeg').convert("L"))
            #print(resulttemp)
            dcttemp = dct( dct( resulttemp, axis=0), axis=1)
            
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
    savefile = open(PATH_GEI + '/trained-model.pkl', 'wb')
    pickle.dump(clf, savefile)
    savefile.close()
    TRAINING_STATUS = False




