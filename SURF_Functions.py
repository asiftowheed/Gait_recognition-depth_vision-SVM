# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:47:23 2019

@author: Asif Towheed
"""


import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct

from skimage import io
import matplotlib.pyplot as plt
from sklearn import svm
import pyrealsense2 as rs
import cv2
from PIL import Image
from PIL import ImageDraw
import pickle


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
        
    def getfeaturevec(self):
        xcounts, xbins, xbars = plt.hist(self.xdelta, range = self.range, density=self.density, rwidth = self.rwidth, bins=self.bins)
        ycounts, ybins, ybars = plt.hist(self.ydelta, range = self.range, density=self.density, rwidth = self.rwidth, bins=self.bins)
        self.FVec = np.append(xcounts, ycounts, axis = 0)
        return self.FVec
        
    def loadhist():
        infile = open('histogram.pkl','rb')
        loaded = pickle.load(infile)
        return loaded
        
    def savehist(self):
        savefile = open('histogram.pkl', 'wb')
        pickle.dump(self, savefile)
        savefile.close()

        

HISTOGRAM = SURF_Histogram()




print(PATH)
print(PATH_SURF)

################################################################################
## BEGIN TRAINING FOR NEW SUBJECT --> CREATE A DIRECTORY WITH THEIR NAME
################################################################################
def SURF_Begin(subjectname):
    
    global SUBJECTNAME, SUBJECTPATH
    SUBJECTNAME = subjectname
    
    print(SUBJECTNAME)
    print(PATH_SURF)
    SUBJECTPATH = PATH_SURF + '/' + SUBJECTNAME
    print(PATH_SURF + '/' + SUBJECTNAME)
    
    try:
        os.mkdir(SUBJECTPATH)
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')
##------------------------------------------------------------------------------- (DONE)
    





################################################################################
## GET THE WALK --> SAVE THE ORIGINAL FRAMES
################################################################################
def SURF_GetWalk(index):
    global INDEX, SUBJECTPATH, STATUSBOOL, STATUSARR
    INDEX = index
    try:
        #print('trying1')
        global FINALPATH
        #print('trying2')
        FINALPATH = SUBJECTPATH + '/WALK' + str(INDEX)
        #print('trying3')
        os.makedirs(FINALPATH)
        #print('trying4')
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')
    
    # Capture the images and store them in 'newSubject/WALK(index)/ORIGINAL
    
    print("Started")
    
    
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
            
            #gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
            #depthwhite = detect2(gray, aligned_depth_frame, clipping_distance*depth_scale) # We get the output of our detect function.
            
            im = Image.fromarray(bg_removed)
            imnumber += 1
            #im.save(FINALPATH + '/' + "{}.jpeg".format(imnumber))
            cv2.imshow('Align Example', bg_removed)
            print('KT', str(KILLTHREAD))
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q") or KILLTHREAD: # Exit condition
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


##------------------------------------------------------------------------------- (DONE)






################################################################################
## GET AN INDIVIDUAL DIFFERENCE IMAGE
################################################################################
def SURF_FindMatches(kp1, kp2, desc1, desc2, n):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    matches = bf.match(desc1,desc2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    #matches[len(matches)-1].distance --> will give the greatest displacement
    newmatches = []
    
    # Error correction --> only consider the points with a correct match
    for match in matches:
        if abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1]) < 20:
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
        a = kp1[i].pt[0]/(xmax1 - xmin1)
        b = kp1[i].pt[1]/(ymax1 - ymin1)
        pic1points.append((a,b))
    
    #print('pic2idxs',pic2idxs)
    #print('kp2', kp2)
    #print('len(kp2)', len(kp2))
    #print(kp2[i].pt[0])
    #print(kp2[i].pt[1])
    for i in pic2idxs:
        #print('i', i)
        a = kp2[i].pt[0]/(xmax2 - xmin2)
        b = kp2[i].pt[1]/(ymax2 - ymin2)
        pic2points.append((a,b))
    
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
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def SURF_GenerateHistogram(): 

    print('---------GENERATING HISTOGRAMS--------------------------------------------')
    
    global NUMPYLIST
    NUMPYLIST = []

    for filename in os.listdir(FINALPATH):
        print(filename)
        openednumpyImage = cv2.imread(FINALPATH + '/' + str(filename))
        NUMPYLIST.append(openednumpyImage)
        
    global STATUSBOOL
    STATUSBOOL = True
    
    surf = cv2.ORB_create(nfeatures = 10)
            
    for i in range(len(NUMPYLIST) - 1):
        print(i, 'called')
        
        kp1, desc1 = surf.detectAndCompute(NUMPYLIST[i], None)
        kp2, desc2 = surf.detectAndCompute(NUMPYLIST[i+1], None)
        
        #i0 = cv2.drawKeypoints(NUMPYLIST[i], kp, None, (255,0,0), 4)
        #i1 = cv2.drawKeypoints(NUMPYLIST[i+1], kp2, None, (255,0,0), 4)
        
        xdelta, ydelta = SURF_FindMatches(kp1, kp2, desc1, desc2, i)
        
        SURF_AddToHistograms(xdelta, ydelta)
        
    fvec = HISTOGRAM.getfeaturevec()
    print(fvec)
##------------------------------------------------------------------------------- (DONE)

    
    



    

################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def SURF_GetStatus():
    global IMAGELIST, NUMPYLIST, STATUSBOOL, FINALDIFFPATH, STATUSARR
    
    if not STATUSBOOL:
        return 0
    else:
        len(STATUSARR)/(len(IMAGELIST)-1) * 20
        return int(len(STATUSARR)/(len(IMAGELIST)-1) * 20)
##-------------------------------------------------------------------------------

    
    



    
    
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
def SURF_RESET():
    global PATH, PATH_AD, SUBJECTNAME, SUBJECTPATH, FINALPATH, INDEX, XDELTA, YDELTA, STATUSBOOL, STATUSARR, KILLTHREAD, width, height
    PATH = os.getcwd().replace('\\','/')
    PATH_AD = PATH + '/SURF'
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
    print('RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------
################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def MID_SURF_RESET():
    global FEATUREVEC, KILLTHREAD, STATUSBOOL, STATUSARR

    FEATUREVEC = []
    KILLTHREAD = False
    STATUSBOOL = False
    STATUSARR = []

    print('MID-RESET CALLED', str(KILLTHREAD))
##-------------------------------------------------------------------------------
