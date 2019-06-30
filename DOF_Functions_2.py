# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:06:01 2019

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
import cv2
import sys
from os.path import isfile, join
from PIL import ImageEnhance
import video

LABELS = []
LABELS_ALL = []
TRAINING_STATUS = False

PATH = os.getcwd().replace('\\','/')
PATH_DOF = PATH + '/DOF'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALORIGINALPATH = ""
FINALDIFFPATH = ""
INDEX = 0

width, height = (640, 480)

IMAGELIST = []
NUMPYLIST = []
ACC_DIFF_IMG = np.zeros([height, width, 3])
FEATUREVEC = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []
PAUSE = False

print(PATH)
print(PATH_DOF)




class DOF_Histogram:
    
    def __init__(self):
        self.ddelta = []
        self.range = (1,30)
        self.bins = 20
        self.density = True
        self.rwidth = 0.90
        self.FVec = []
        
    def addd(self, ddelta):
        self.ddelta += ddelta
        
    def resetd(self):
        self.ddelta = []

    def resetfvec(self):
        self.FVec = []
        
    def getfeaturevec(self):
        dcounts, dbins, dbars = plt.hist(self.ddelta, range = self.range, density=self.density, rwidth = self.rwidth, bins=self.bins)
        self.FVec = dcounts
        return self.FVec
        
    def loadhist(location):
        infile = open(location + '/fvec_dof.pkl','rb')
        loaded = pickle.load(infile)
#        print(loaded)
        return loaded

    def loadfvec(location):
        infile = open(location + '/fvec_dof.pkl','rb')
        loaded = pickle.load(infile)
        return loaded
        
    def savehist(self, location, data):
        print('saving AccD')
#        savefile = open(location + '/AD.pkl', 'wb')
        savefile2 = open(location + '/fvec_dof.pkl', 'wb')
#        pickle.dump(self, savefile)
        pickle.dump(data, savefile2)
        savefile2.close()        

HISTOGRAM = DOF_Histogram()





################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def DOF_Begin():
    
    print('DOF_Begin()')
    
    ## Getting Directories
    ##=============================================================================
    global LABELS, LABELS_ALL
    PATH = os.getcwd().replace('\\', '/')
#    print('PATH:\t\t', PATH)
    
    DOF_PATH = PATH + '/DOF'
#    print('AD_PATH:\t', AD_PATH)
    
    for filename in os.listdir(DOF_PATH):
        #print('--------------')
#        if (re.match('.*_W', filename)):
#            LABELS_W.append(filename)
        if re.match('.*_untrained', filename):
            LABELS.append(filename)
        elif not re.match('trained-model', filename):
            LABELS.append(filename)
            
            
    # Training on frames
    #=============================================================================
    
    for i in LABELS:                                          # open each recorded person
        global SUBJECTNAME, KILLTHREAD
        SUBJECTNAME = i
        if KILLTHREAD:
            return

        #----------------------------------------------------------------------
        DOF_Generate_AMV(SUBJECTNAME)
        #----------------------------------------------------------------------

        ts = time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + '\n'
        file = open(PATH + '/history.txt', 'r+')
        contents_as_string = file.read()
        newtext = SUBJECTNAME.split('_untrained')[0] + ' was trained for AMV: ' + timestamp + contents_as_string
        file2 = open(PATH + '/history.txt', 'w')
        file2.write(newtext)
        file.close()
        file2.close()

    #-----------------------------------------------------------------------------
##-------------------------------------------------------------------------------





################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #remove the last file which is a videofile
    if(files[-1] == 'video.avi'):
        return
    else:
        files = files[:-1]
     
        #for sorting the file names properly in the format ('n.jpeg')
        #n is the frame number
    files.sort(key = lambda x: int(x[0:-5]))


    for i in range(len(files)):
        filename=pathIn + '/' + files[i]
        #reading each files
        im = Image.open(filename)
        enhancer = ImageEnhance.Contrast(im)
        e_im = enhancer.enhance(5)
        e_im.save(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
#        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, (640,480))
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

##------------------------------------------------------------------------------- (DONE)
    
    

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv0 = ang*(180/np.pi/2)
    hsv2 = np.minimum(v*4, 255)
    hsv[...,0] = hsv0
#    hsv[...,1] = 255
    hsv[...,2] = hsv2
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr




def getStatus2(timeinit):
    tfinal = time()
    global TRAINING_STATUS
    if not TRAINING_STATUS:
        return str(tfinal-timeinit) + 's; Processing walk ' + str(INDEX) + ' for ' + str(SUBJECTNAME) + '...'
    else:
        return 'Training the classifier...'


def DOF_GetWalkNumber():
    return int(INDEX)


################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def DOF_GetStatus():
    global IMAGELIST, NUMPYLIST, STATUSBOOL, FINALDIFFPATH, STATUSARR, PAUSE
    
    if not STATUSBOOL:
        return 0
    elif PAUSE:
        return 20
    else:
        #print('len(STATUSARR)', len(STATUSARR))
        #print('(len(IMAGELIST)-1)', (len(IMAGELIST)-1))
        #print('len(STATUSARR)/(len(IMAGELIST)-1) * 20',len(STATUSARR)/(len(IMAGELIST)-1) * 20)
        return int(math.ceil(len(STATUSARR)/(len(os.listdir(FINALORIGINALPATH))-2) * 20))
##-------------------------------------------------------------------------------




def DOF_Generate_AMV(label):
    walks = []                                              # array to store how many times they have walked
    label_path = PATH_DOF + '/' + label
    combine = np.zeros((480,640,3))                          # path to that person
    SUBJECTPATH = PATH_DOF + '/' + label
    for walk in os.listdir(label_path):

        global STATUSBOOL, STATUSARR, INDEX, FINALORIGINALPATH

        FINALORIGINALPATH = SUBJECTPATH + '/' + walk + '/ORIGINAL'
        INDEX = walk[-1]
        print('DOF for ' + str(walk) + ' for ' + str(label))
        walks.append(walk)                                  # get each walk
        walk_path = label_path + '/' + walk + '/ORIGINAL'   # path to that walk
        
        pathIn= walk_path
        pathOut = walk_path + '/video.avi'
        fps = 20.0
        #check if the video already exists
        convert_frames_to_video(pathIn, pathOut, fps)
        counter = 0
        
        STATUSBOOL = False
        STATUSARR = []        

        
        if(os.path.isfile(label_path + '/' + walk +'/fvec_dof.pkl')):
            print ('---------------File already saved, skipping------------')
        else:
            print('-------------------GENERATING DCTs------------------------')
            try:
                #this is the video that is created
                fn = pathOut
            except IndexError:
                fn = 0
    
            cam = video.create_capture(fn)
            ret, prev = cam.read()
            prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            while True:
                counter += 1
                ret, img = cam.read()
                if(not ret):
                    break
                if(counter%1 == 0):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
                    prevgray = gray
    #            draw_flow(gray,flow)
                    bgr = draw_hsv(flow)
                    combine = np.add(combine, bgr)
                    STATUSARR.append(1)
            print (label_path + '/' + walk)
            HISTOGRAM.savehist(label_path + '/' + walk, combine)
    
        
                
#        print(fvec)
        print('---------------------Walk finished-----------------------')
#        HISTOGRAM.resetd()
#        HISTOGRAM.resetfvec()
#    os.rename(SUBJECTPATH, SUBJECTPATH.split('_untrained')[0])
    LABELS_ALL.append(label)
    MID_DOF_RESET()
##-----------------------------------------------------------------------------


def DOF_fvecs():
    first = True
    second = True
    traininglabels = []
    mainvec = []
    firstvec = None
    secondvec = None
    for i in LABELS_ALL:                                          # open each walk for recorded person
        walks = []                                              # array to store how many times they have walked
        label_path = PATH_DOF + '/' + i                          # path to that person        
        
        
        for walk in os.listdir(label_path):
#            -- code for cross testing --> TRAIN ON JEANS
#            if os.path.exists('C:/Users/Asif Towheed/Desktop/Trimmed/K_A/' + i + '/' + walk):
#                break
#            -- code for cross testing --> TRAIN ON K/A
            if not os.path.exists('C:/Users/Asif Towheed/Desktop/Trimmed/K_A/' + i + '/' + walk):
                continue
            walks.append(walk)                                  # get each walk
            walk_path = label_path + '/' + walk                 # path to that walk
            AD = DOF_Histogram.loadhist(walk_path)
#            print(AD)
            dcttemp = dct( dct( AD, axis=0), axis=1)
            AD = np.array(AD)
#            FVec = FVec.reshape(1,-1)
            featurevec = []
            
            for m in range(10):
                for n in range(10):
                    featurevec.append(dcttemp[m][n])
            
            FVec = np.array(featurevec)
            FVec = FVec.reshape(1,-1)
            
            
#            global first, second, firstvec, secondvec, mainvec
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
    savefile = open(PATH_DOF + '/trained-model.pkl', 'wb')
    pickle.dump(clf, savefile)
    savefile.close()






################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def DOF_RESET():
    global SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, KILLTHREAD, STATUSARR, STATUSBOOL
    SUBJECTNAME = ""
    SUBJECTPATH = ""
    FINALORIGINALPATH = ""
    FINALDIFFPATH = ""
    INDEX = 0
    STATUSARR = []
    STATUSBOOL = False
        
    KILLTHREAD = False
    print('RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------
################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def MID_DOF_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, INDEX, IND_DIFF_IMG, ACC_DIFF_IMG, FEATUREVEC, KILLTHREAD, STATUSBOOL, STATUSARR

    KILLTHREAD = False
    STATUSBOOL = False
    STATUSARR = []

    print('MID-RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------

################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def TerminateDifferences():
    global KILLTHREAD
    KILLTHREAD = True
##-------------------------------------------------------------------------------








