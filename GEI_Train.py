# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:19:38 2019

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
import math
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  


width, height = (640, 480)



PATH = os.getcwd().replace('\\','/')
PATH_GEI = PATH + '/GEI'
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


PADDEDPATH = None
PREPATH = None

width, height = (640, 480)
ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)


################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def GEI_FindDiff(im1, im2, n, walk_path):
    global ACC_DIFF_IMG
    IND_DIFF_IMG = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    sumofdiff = 0
    
    for w in range(width):
        for h in range(height):
            absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
            #print("PRINTING", int(im1[h,w]))
            if absdiff < 30:
                absdiff = 0
            sumofdiff += absdiff
            IND_DIFF_IMG[h,w] = absdiff
    
    newthresh = sumofdiff/(width*height)
    
    for w in range(width):
        for h in range(height):
            if IND_DIFF_IMG[h,w] < newthresh:
                IND_DIFF_IMG[h,w] = 0
            else:
                ACC_DIFF_IMG[h,w] += 10
    print("SAVING" + walk_path)
    newimage = Image.fromarray(IND_DIFF_IMG)
    print
    newimage.save(walk_path + "/DIFFERENCES/IND-IMAGE-{}.jpeg".format(n))
    #STATUSARR.append(n)


##------------------------------------------------------------------------------- (DONE)
    

################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def GEI_GenerateGEI(FINALORIGINALPATH, PADDEDPATH, PREPATH):

#    try:
#        global FINALDIFFPATH
#        FINALDIFFPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/DIFFERENCES"
#        os.mkdir(FINALDIFFPATH)
#    except OSError:
#        print('Error in creation of directory')
#    else:
#        print('Successfully created directory')    

    print('---------EXTRACTING IMAGES INTO AN ARRAY--------------------------------------------')
    
    global IMAGELIST, NUMPYLIST
    IMAGELIST = []
    NUMPYLIST = []

    sumw = sumh = maxw = maxh = 0
    count = 0
    for filename in os.listdir(PREPATH):
        print(filename)
#        openedImage = Image.open(FINALORIGINALPATH + '/' + str(filename)).convert("L")
#        IMAGELIST.append(openedImage)
#        NUMPYLIST.append(np.array(openedImage))
        #                 for filepath in glob.iglob(PATH + '/*.jpeg'):
        #path = PATH + '/*.jpeg'
#        print(filepath)
#        print(path)

        if re.match('Pre-processed_.*',filename):
            print(filename)
            ori = cv2.imread(PREPATH + '/' + filename)
            height, width, depth = ori.shape
            sumw += width
            sumh += height
            if width > maxw:
                maxw = width
            if height > maxh:
                maxh = height
            print(width, height)
            count += 1

    avgw = sumw/count
    avgh = sumh/count
    print('count', count)
    result = None
    checker = True

    npimages = []


    for filename in os.listdir(PREPATH):
        if re.match('Pre-processed_.*',filename):
            ori = cv2.imread(PREPATH + '/' + filename)
            height, width, depth = ori.shape
            print(width, height)
            print('---',filename,'---')
            if width < 150 or width > 300:
                print('removing', filename)
                #os.unlink(filename)
            else:
                print('adding', filename)
                #newimg = cv2.resize(ori,(int(maxw),int(maxh)))
                to_be_added_w = (640-width)
                to_be_added_h = (480-height)
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

    print('npimages:', npimages)
    print('lenth of npimages:', len(npimages))
    for i in range(len(npimages)-1):
        #newimg = finddiff(npimages[i], npimages[i+1])
        #h1, w1 = np.array(result).size
        #h2, w2 = newimg.size
        #print("h1",h1,"w1",w1," h2",h2,"w2",w2)
        #print(result == None)
        print('-----------------------------')
        pilimg = Image.fromarray(np.array(result))
        pilimg.save(PADDEDPATH + '/result.jpeg')
        pilimg = Image.fromarray(np.array(npimages[i]))
        pilimg.save(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        newimg = cv2.imread(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        result = cv2.imread(PADDEDPATH + '/result.jpeg')
        
        print(PADDEDPATH + '/Padded_{}.jpeg'.format(i))
        result = cv2.addWeighted(result, 0.9, newimg, 0.1, 0)    
       
    global STATUSBOOL
    STATUSBOOL = True
            
#    for i in range(len(NUMPYLIST) - 1):
#        print(i, 'called')
#        GEI_FindDiff(NUMPYLIST[i], NUMPYLIST[i+1], i)

##-------------------------------------------------------------------------------


################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def GEI_Preprocess(FINALORIGINALPATH, PADDEDPATH, PREPATH):

    print('GEI_Preprocess foP:', FINALORIGINALPATH)
    for filename in os.listdir(FINALORIGINALPATH):
        print('FILENAMING', filename)
        imagearr = np.array(Image.open(FINALORIGINALPATH + '/' + str(filename)).convert("L"))
        lowest = rightmost = 1
        topmost = 480
        leftmost = 640
        detected = False
        imagearr2 = np.copy(imagearr)
    
        for y in range(480):
            for x in range(640):
            
                if imagearr2[y][x] != 0:
                    imagearr2[y][x] = 255
                    if y < topmost:
                        topmost = y
                        detected = True
                    if y > lowest:
                        lowest = y
                        detected = True
                    if x > rightmost:
                        rightmost = x
                        detected = True
                    if x < leftmost:
                        leftmost = x
                        #print(y)
                        #print(ypos,"\n")
                        detected = True
   
    #    print("DONE\n\n\nDONE\n\n\nDONE\n\n\nDONE\n\n\nDONE\n\n\n")
        print("top", topmost)
        print("lowest", lowest)
        print("left", leftmost)
        print("right", rightmost)
        if detected is True and leftmost != rightmost and topmost != lowest:
            bbox = (leftmost, topmost, rightmost, lowest)
            print(bbox)
            #person = imagearr[topmost:lowest-topmost, leftmost:rightmost-leftmost]
            pilimg = Image.fromarray(imagearr2).crop(bbox)
    #        print("saving")
#            global l
#            l = l + 1
    #        pilimg.show()
            pilimg.save(PREPATH + '/Pre-processed_' + filename)
            print('saved preprocessed ' + filename + ' at ' + PREPATH)


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



## Getting Directories
##=============================================================================
LABELS = []
LABELS_W = []
PATH = os.getcwd().replace('\\', '/')
print('PATH:\t\t', PATH)

GEI_PATH = PATH + '/GEI'
print('GEI_PATH:\t', GEI_PATH)

for filename in os.listdir(PATH_GEI):
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

def GEI_fvecs():
    for j in range(40):                                          # open each depthwhite recorded person
        i = LABELS[j]
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_GEI + '/' + i                          # path to that person
        
        
        if not re.match('trained',i):        
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
                for m in range(50):
                    for n in range(50):
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
#    clf = svm.SVC(kernel='linear', C = 1)
#    clf.fit(mainvec, traininglabels)
#    savefile = open(location + '/histogram.pkl', 'wb')
#    pickle.dump(self, savefile)
#    savefile.close()

GEI_fvecs()
#
tFinal = time()
print("Linear SVM training time = ", round(tFinal-tInit, 5), "s")
#
##
#
#clf = svm.SVC(kernel='linear', C = 1)
#clf.fit(mainvec, traininglabels)
#
#
#imtest = np.array(Image.open(r'C:\Users\Asif Towheed\Desktop\Trimmed\GEI\PAJI\WALK2\PADDED\result.jpeg').convert("L"))
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


















