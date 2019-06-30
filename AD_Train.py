# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:37:37 2019

@author: Asif Towheed
"""

from sklearn import svm
from PIL import Image
import os
import re
import numpy as np
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


width, height = (640, 480)
ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
##
##
################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def AD_FindDiff(im1, im2, n, walk_path):
    
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
    newimage.save(walk_path + "/DIFFERENCES/IND-IMAGE-{}.jpeg".format(n))
    #STATUSARR.append(n)

##-------------------------------------------------------------------------------
##
##
##
##
##
##
##
##
## Getting Directories
##=============================================================================
LABELS = []
LABELS_W = []
PATH = os.getcwd().replace('\\', '/')
print('PATH:\t\t', PATH)

AD_PATH = PATH + '/AD'
print('AD_PATH:\t', AD_PATH)

for filename in os.listdir(AD_PATH):
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
#
#for i in LABELS_W:                                          # open each depthwhite recorded person
#    walks = []                                              # array to store how many times they have walked
#    label_path = AD_PATH + '/' + i                          # path to that person
#    for walk in os.listdir(label_path):
#        walks.append(walk)                                  # get each walk
#        walk_path = label_path + '/' + walk                 # path to that walk
#        
#        IMAGELIST = []
#        NUMPYLIST = []
#        for frame in os.listdir(walk_path + '/ORIGINAL'):
#            openedImage = Image.open(walk_path + '/ORIGINAL/' + frame).convert("L")
#            #print(walk_path + '/ORIGINAL/' + frame)
#            IMAGELIST.append(openedImage)
#            NUMPYLIST.append(np.array(openedImage))
#        
#        os.mkdir(walk_path + '/DIFFERENCES')
#        
#        k = 0
#        for j in range(len(NUMPYLIST) - 1):
#            k += 1
#            print()
#            AD_FindDiff(NUMPYLIST[j], NUMPYLIST[j+1], k, walk_path)
#        
#        newimage = Image.fromarray(ACC_DIFF_IMG)
#        newimage.save(walk_path + "/AD.jpeg")
#        ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
#        
#        print('AD Complete for ' + walk + ' for ' + i)
#
#    print(i, walks)
##-----------------------------------------------------------------------------



### Training on Depthwhite frames
###=============================================================================

#for i in LABELS:                                          # open each depthwhite recorded person
#    if re.match('ZEENAH', i):
#        walks = []                                              # array to store how many times they have walked
#        label_path = AD_PATH + '/' + i                          # path to that person
#        for walk in os.listdir(label_path):
#            walks.append(walk)                                  # get each walk
#            walk_path = label_path + '/' + walk                 # path to that walk
#            
#            IMAGELIST = []
#            NUMPYLIST = []
#            for frame in os.listdir(walk_path + '/ORIGINAL'):
#                openedImage = Image.open(walk_path + '/ORIGINAL/' + frame).convert("L")
#                #print(walk_path + '/ORIGINAL/' + frame)
#                IMAGELIST.append(openedImage)
#                NUMPYLIST.append(np.array(openedImage))
#            
#            if not os.path.exists(walk_path + '/DIFFERENCES'):
#                os.mkdir(walk_path + '/DIFFERENCES')
#
#            
#            k = 0
#            for j in range(len(NUMPYLIST) - 1):
#                k += 1
#                AD_FindDiff(NUMPYLIST[j], NUMPYLIST[j+1], k, walk_path)
#            
#            newimage = Image.fromarray(ACC_DIFF_IMG)
#            newimage.save(walk_path + "/AD.jpeg")
#            ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
#            
#            print('AD Complete for ' + walk + ' for ' + i)
#    
#        print(i, walks)
###-----------------------------------------------------------------------------
        
def checkPlain(im):
    count = 0
    for w in range(width):
        for h in range(height):
            if int(im[h,w]) > 0:
                count += 1
                if count >= 10000:
                    return False
    return True
        
## Training on Depthwhite frames
##=============================================================================

#for i in LABELS:                                          # open each depthwhite recorded person
#    walks = []                                              # array to store how many times they have walked
#    label_path = AD_PATH + '/' + i                          # path to that person
#    for walk in os.listdir(label_path):
#        walks.append(walk)                                  # get each walk
#        walk_path = label_path + '/' + walk                 # path to that walk
#        
#        IMAGELIST = []
#        NUMPYLIST = []
#        for frame in os.listdir(walk_path + '/ORIGINAL'):
#            openedImage = Image.open(walk_path + '/ORIGINAL/' + frame).convert("L")
#            #print(walk_path + '/ORIGINAL/' + frame)
#            npimg = np.array(openedImage)
#            
#            if checkPlain(npimg):
#                os.remove(walk_path + '/ORIGINAL/' + frame)
#        
#                
#        print('Clearing Complete for ' + walk + ' for ' + i)
#
#    print(i, walks)
##-----------------------------------------------------------------------------


first = True
second = True
traininglabels = []
mainVec = None
firstvec = None
secondvec = None

def DCT_fvecs():
    for j in range(len(LABELS)):                                          # open each depthwhite recorded person
        i = LABELS[j]
        walks = []                                              # array to store how many times they have walked
        label_path = AD_PATH + '/' + i                          # path to that person
        
        
        if not re.match('trained',i):
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

                

    
        print(i, walks)

DCT_fvecs()
#

#clf = svm.SVC(kernel='linear', C = 1.0, probability = True)
#clf.fit(mainvec, traininglabels)
#
#
##y_pred = clf.predict(featurevecsw4)
##print(y_pred)
#
#
#imtest = np.array(Image.open(r'C:\Users\Asif Towheed\Documents\DEVEL\Gait_Recognition\REC\Test-Walk-1556467200\AD.jpeg').convert("L"))
#
#dcttest = dct( dct( imtest, axis=0), axis=1)
#
#featurevec2 = []
#for m in range(10):
#    for n in range(10):
#        featurevec2.append(dcttest[m][n])
#
#
#fvectest = np.array(featurevec2)
#
#fvectest = fvectest.reshape(1,-1)
#
#y_pred = clf.predict(fvectest)
#print(y_pred)
#y_pred = clf.predict_proba(fvectest)
#print(y_pred)
#
#maxprob = 0
#for i in y_pred:
#    for j in i:
#        if j > maxprob:
#            maxprob = j
#        
#if maxprob < 0.1:
#    print('Not trained.')
#else:
#    y_pred = clf.predict(fvectest)
#    print(y_pred)
#    


X_train, X_test, y_train, y_test = train_test_split(mainvec, traininglabels, test_size = 0.20)  

svclassifier = svm.SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)  

cm = confusion_matrix(y_test,y_pred)
print(cm)  
cr = classification_report(y_test,y_pred)
print(cr)
score = accuracy_score(y_test, y_pred)
print(score)











