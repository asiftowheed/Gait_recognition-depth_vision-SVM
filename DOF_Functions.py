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
import math
import pyrealsense2 as rs

from skimage import io
from sklearn import svm
from PIL import ImageDraw
import pickle
from PIL import ImageEnhance
from os.path import isfile, join

import video

width, height = (640, 480)

PATH = os.getcwd().replace('\\','/')
#PATH_DOF = PATH + '/DOF'
PATH_DOF = PATH + '/DOF'
SUBJECTNAME = ""
SUBJECTPATH = ""
FINALPATH = ""
INDEX = 0

width, height = (640, 480)

DDELTA = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []
STATUSBOOL = False

print(PATH)
print(PATH_DOF)


tInit = time()

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
        infile = open(location + '/fvec.pkl','rb')
        loaded = pickle.load(infile)
#        print(loaded)
        return loaded

    def loadfvec(location):
        infile = open(location + '/fvec.pkl','rb')
        loaded = pickle.load(infile)
        return loaded
        
    def savehist(self, location, data):
        print('saving AccD')
#        savefile = open(location + '/AD.pkl', 'wb')
        savefile2 = open(location + '/fvec.pkl', 'wb')
#        pickle.dump(self, savefile)
        pickle.dump(data, savefile2)
        savefile2.close()        

HISTOGRAM = DOF_Histogram()





################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def DOF_Begin(subjectname):
    
    global SUBJECTNAME, SUBJECTPATH
    SUBJECTNAME = subjectname
    
    print(SUBJECTNAME)
    print(PATH_DOF)
    SUBJECTPATH = PATH_DOF + '/' + SUBJECTNAME
    print(PATH_DOF + '/' + SUBJECTNAME)
    
    try:
        os.mkdir(SUBJECTPATH)
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')
##-------------------------------------------------------------------------------





################################################################################
## GET THE WALK --> SAVE THE ORIGINAL FRAMES
################################################################################
def DOF_GetWalk(index):
    global INDEX, SUBJECTPATH, STATUSBOOL, STATUSARR
    INDEX = index
    try:
        print('trying1')
        global FINALORIGINALPATH
        print('trying2')
        FINALORIGINALPATH = SUBJECTPATH + '/WALK' + str(INDEX) + "/ORIGINAL"
        print('trying3')
        os.makedirs(FINALORIGINALPATH)
        print('trying4')
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')
    
    # Capture the images and store them in 'newSubject/WALK(index)/ORIGINAL
    
    print("Started")
    
    getframe = False
    
    #################################################################################
    # Create a pipeline
    pipeline = rs.pipeline()
    
    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    #################################################################################
    
    
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3 #5.5 meter optimum for gait recognition
    clipping_distance = clipping_distance_in_meters / depth_scale
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Streaming loop
    try:
        imnumber = 0
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 0
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            
            gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
            #depthwhite = detect2(gray, aligned_depth_frame, clipping_distance*depth_scale) # We get the output of our detect function.
            
            im = Image.fromarray(gray)
            #im = Image.fromarray(depthwhite)
            imnumber += 1
            im.save(FINALORIGINALPATH + '/' + "{}.jpeg".format(imnumber))
            #cv2.imshow('Align Example', gray)
            cv2.imshow('Align Example', gray)
            global KILLTHREAD
            print('KT', str(KILLTHREAD))
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q") or KILLTHREAD: # Exit condition
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


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



### Getting Directories
###=============================================================================
#LABELS = []
#LABELS_W = []
#PATH = os.getcwd().replace('\\', '/')
#print('PATH:\t\t', PATH)
#
#for filename in os.listdir(PATH_DOF):
#    #print('--------------')
#    if (re.match('.*_W', filename)):
#        LABELS_W.append(filename)
#    else:
#        LABELS.append(filename)
    
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

#for label in LABELS:                                          # open each depthwhite recorded person
#    walks = []                                              # array to store how many times they have walked
#    label_path = PATH_DOF + '/' + label
#    combine = np.zeros((480,640,3))                          # path to that person
#    for walk in os.listdir(label_path):
#        print('DOF for ' + str(walk) + ' for ' + str(label))
#        walks.append(walk)                                  # get each walk
#        walk_path = label_path + '/' + walk + '/ORIGINAL'   # path to that walk
#        
#        pathIn= walk_path
#        pathOut = walk_path + '/video.avi'
#        fps = 20.0
#        #check if the video already exists
#        convert_frames_to_video(pathIn, pathOut, fps)
#        counter = 0
#        if(os.path.isfile(label_path + '/' + walk +'/fvec.pkl')):
#            print ('---------------File already saved, skipping------------')
#        else:
#            print('-------------------GENERATING DCTs------------------------')
#            import sys
#            try:
#                #this is the video that is created
#                fn = pathOut
#            except IndexError:
#                fn = 0
#    
#            cam = video.create_capture(fn)
#            ret, prev = cam.read()
#            prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#            while True:
#                counter += 1
#                ret, img = cam.read()
#                if(not ret):
#                    break
#                if(counter%1 == 0):
#                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
#                    prevgray = gray
#    #            draw_flow(gray,flow)
#                    bgr = draw_hsv(flow)
#                    combine = np.add(combine, bgr)
#    #        print (combine)
#            HISTOGRAM.savehist(label_path + '/' + walk, combine)
#    
#        
#                
##        print(fvec)
#        print('---------------------Walk finished-----------------------')
##        HISTOGRAM.resetd()
##        HISTOGRAM.resetfvec()
###-----------------------------------------------------------------------------

        
def checkPlain(im):
    count = 0
    for w in range(width):
        for h in range(height):
            if int(im[h,w]) > 0:
                count += 1
                if count >= 5000:
                    return False
    return True
        



#def DOF_fvecs():
#    first = True
#    second = True
#    traininglabels = []
#    mainvec = []
#    firstvec = None
#    secondvec = None
#    for i in LABELS:                                          # open each walk for recorded person
#        walks = []                                              # array to store how many times they have walked
#        label_path = PATH_DOF + '/' + i                          # path to that person        
#        
#        
#        for walk in os.listdir(label_path):
#            walks.append(walk)                                  # get each walk
#            walk_path = label_path + '/' + walk                 # path to that walk
#            AD = DOF_Histogram.loadhist(walk_path)
##            print(AD)
#            dcttemp = dct( dct( AD, axis=0), axis=1)
#            AD = np.array(AD)
##            FVec = FVec.reshape(1,-1)
#            featurevec = []
#            
#            for m in range(10):
#                for n in range(10):
#                    featurevec.append(dcttemp[m][n])
#            
#            FVec = np.array(featurevec)
#            FVec = FVec.reshape(1,-1)
#            
#            
#            global first, second, firstvec, secondvec, mainvec
#            if first and second:
#                firstvec = FVec
#                first = False
#            elif second:
#                secondvec = FVec
#                mainvec = np.append(firstvec, secondvec, axis = 0)
#                second = False
#            else:
#                mainvec = np.append(mainvec, FVec, axis = 0)
#            traininglabels.append(i)
#                
#        print(i, walks)
        
#DOF_fvecs()

tFinal = time()
print("Linear SVM training time = ", round(tFinal-tInit, 5), "s")


##-----------------------for testing a single walk-----------------------------

#clf = svm.SVC(kernel='linear', C = 1.0)
#clf.fit(mainvec, traininglabels)
#
#
##y_pred = clf.predict(featurevecsw4)
##print(y_pred)
#
##3 metres is 9  lines in EB2
#
#adtest = DOF_Histogram.loadhist(r'E:\Gait_Recognition\testfolder2\asifrecog5\WALK2')
#dcttest = dct( dct( adtest, axis=0), axis=1)
#
#featurevec2 = []
#for m in range(10):
#    for n in range(10):
#        featurevec2.append(dcttest[m][n])
#
#
#fvectest = np.array(featurevec2)
#fvectest = fvectest.reshape(1,-1)
#
#y_pred = clf.predict(fvectest)
#print(y_pred)

#---------------------for calculating confusion matrix-------------------------

#from sklearn.model_selection import train_test_split  
#from sklearn.metrics import classification_report, confusion_matrix  
#
#
#
#
#
#X_train, X_test, y_train, y_test = train_test_split(mainvec, traininglabels, test_size = 0.2)  
#
#svclassifier = svm.SVC(kernel='linear')  
#svclassifier.fit(X_train, y_train)
#
#y_pred = svclassifier.predict(X_test)  
#
#cm = confusion_matrix(y_test,y_pred)
#print(cm)  
#cr = classification_report(y_test,y_pred)
#print(cr)
#


#testvec = DOF_Histogram.loadfvec(r'C:\Users\wasif\OneDrive\Documents\GitHub\GaitRecog\testfolder2\asif\WALK2')
#testvec = np.array(testvec)
#testvec = testvec.reshape(1,-1)
#
#y_pred = clf.predict(testvec)
#print(y_pred)



################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def TerminateCapture():
    global KILLTHREAD
    KILLTHREAD = True
##-------------------------------------------------------------------------------







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


















