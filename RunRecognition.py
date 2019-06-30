# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:02:38 2019

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
from PIL import Image, ImageEnhance
from PIL import ImageDraw
import re
import math
import pickle
import time
from os.path import isfile, join
import video


PATH = os.getcwd().replace('\\','/')
PATH_REC = PATH + '/REC'
AD_PATH = PATH_REC + '/AD'
GEI_PATH = PATH_REC + '/GEI'
SURF_PATH = PATH_REC + '/SURF'
DOF_PATH = PATH_REC + '/DOF'
SELECTEDALG = ""


SUBJECTNAME = ""
SUBJECTPATH = ""
FINALORIGINALPATH = ""
FINALDIFFPATH = ""
PADDEDPATH = ""
PREPATH = ""
INDEX = None

width, height = (640, 480)

IMAGELIST = []
NUMPYLIST = []
ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
FEATUREVEC = []
KILLTHREAD = False
STATUSBOOL = False
STATUSARR = []

print(PATH)
print(PATH_REC)





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
## GET THE WALK --> SAVE THE ORIGINAL FRAMES
################################################################################
def RR_GetWalk(SELECTEDALG):
    global INDEX, SUBJECTPATH, STATUSBOOL, STATUSARR, PATH_REC, SUBJECTPATH, FINALORIGINALPATH, FINALDIFFPATH, PADDEDPATH, PREPATH
#    INDEX = index
    try:
        if not os.path.exists(PATH_REC):
            os.mkdir(PATH_REC)
        SUBJECTPATH = PATH_REC + '/Test-Walk-' + str(time.time()).split('.')[0]
        os.mkdir(SUBJECTPATH) #create a test walk folder with the current dat,time in millis as the folder name (splitting until the decimal)
        FINALORIGINALPATH = SUBJECTPATH + '/ORIGINAL'
        os.mkdir(FINALORIGINALPATH)
        FINALDIFFPATH = SUBJECTPATH + '/DIFFERENCES'
        os.mkdir(FINALDIFFPATH)
        PADDEDPATH = SUBJECTPATH + "/PADDED"
        os.mkdir(PADDEDPATH)
        PREPATH = SUBJECTPATH + "/PRE-FOR-GEI"
        os.mkdir(PREPATH)
        
    except OSError:
        print('Error in creation of directory')
    else:
        print('Successfully created directory')
    
    # Capture the images and store them in 'newSubject/WALK(index)/ORIGINAL
    
    print("Started")
    
    global KILLTHREAD
    KILLTHREAD = False
    
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
            imnumber += 1
            im.save(FINALORIGINALPATH + '/' + "{}.jpeg".format(imnumber))
            cv2.imshow('Align Example', gray)
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

        
#    print(xdelta, ydelta)
    return xdelta, ydelta

##------------------------------------------------------------------------------- (DONE)
    







################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def SURF_AddToHistograms(xdelta, ydelta):
    
    global HISTOGRAM
    
    print('before adding',HISTOGRAM.xdelta)
    HISTOGRAM.addx(xdelta)
    print('after adding:',HISTOGRAM.xdelta)
    HISTOGRAM.addy(ydelta)    
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
    print(1111)
    global PREPATH
    print(PREPATH)
#    for filename in os.listdir(PREPATH):
#        print(filename)
##        openedImage = Image.open(FINALORIGINALPATH + '/' + str(filename)).convert("L")
##        IMAGELIST.append(openedImage)
##        NUMPYLIST.append(np.array(openedImage))
#        #                 for filepath in glob.iglob(PATH + '/*.jpeg'):
#        #path = PATH + '/*.jpeg'
##        print(filepath)
##        print(path)
#
#        if re.match('Pre-processed_.*',filename):
#            print(filename, 222)
#            ori = cv2.imread(PREPATH + '/' + filename)
#            height, width, depth = ori.shape
#            sumw += width
#            sumh += height
#            if width > maxw:
#                maxw = width
#            if height > maxh:
#                maxh = height
#            print(width, height)
#            count += 1

    #avgw = sumw/count
    #avgh = sumh/count
    #print('count', count)
    result = None
    checker = True

    npimages = []


    for filename in os.listdir(PREPATH):
        if re.match('Pre-processed_.*',filename):
            ori = cv2.imread(PREPATH + '/' + filename)
            height, width, depth = ori.shape
            print(width, height)
            print('---',filename,'---')
#            if width < 150 or width > 300:
#                dummy = 0
#                #print('removing', filename)
#                #os.unlink(filename)
#            else:
            if not width < 150 and not width > 300:
                to_be_added_w = (640-width)
                to_be_added_h = (480-height)                
                print('adding', filename)
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



def checkPlain(im):
    count = 0
    for w in range(width):
        for h in range(height):
            if int(im[h,w]) > 0:
                count += 1
                if count >= 10000:
                    return False
    return True


def AD_GetWalkNumber():
    return int(INDEX)
    


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


################################################################################
## GET THE DIFFERENCES --> SAVE THE INDIVIDUAL DIFFERENCE IMAGES
################################################################################
def Preprocess():        
    STATUSBOOL = False
    STATUSARR = []        
    IMAGELIST = []
    NUMPYLIST = []
    global ACC_DIFF_IMG
    ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
    num = 0 
    for frame in os.listdir(FINALORIGINALPATH):
        if num%3 == 0:
            openedImage = Image.open(FINALORIGINALPATH + '/' + frame).convert("L")
            #print(walk_path + '/ORIGINAL/' + frame)
            if not checkPlain(np.array(openedImage)):
                IMAGELIST.append(openedImage)
                NUMPYLIST.append(np.array(openedImage))
        num += 1
    
    if not os.path.exists(FINALDIFFPATH):
        os.mkdir(FINALDIFFPATH)

#        STATUSBOOL = True
#        PAUSE = False

    k = 0
    for j in range(len(NUMPYLIST) - 1):
        k += 1
        AD_FindDiff(NUMPYLIST[j], NUMPYLIST[j+1], k)

#    print (ACC_DIFF_IMG)
    newimage = Image.fromarray(ACC_DIFF_IMG)
    newimage.save(SUBJECTPATH + "/AD.jpeg")
    ACC_DIFF_IMG = np.zeros([height, width], dtype=np.uint8)
    
    
    
    print('GEI_Preprocess foP:', FINALORIGINALPATH)
#        num = 0
    for filename in os.listdir(FINALDIFFPATH):
#            if num%3 == 0:
#        print('FILENAMING', filename)
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
    
    
    
    HISTOGRAM.resetx()
    HISTOGRAM.resety()
    HISTOGRAM.resetfvec()
    
    combine = np.zeros((480,640,3))
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
        
        
    pathIn= FINALORIGINALPATH
    pathOut = FINALORIGINALPATH + '/video.avi'
    fps = 20.0
    #check if the video already exists
    convert_frames_to_video(pathIn, pathOut, fps)
    counter = 0
    print('-------------------GENERATING DCTs------------------------')
    import sys
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
#        print (combine)
    savefile2 = open(SUBJECTPATH + '/AMV.pkl', 'wb')
#        pickle.dump(self, savefile)
    pickle.dump(combine, savefile2)
    savefile2.close()
    
    surf = cv2.ORB_create(nfeatures = 100)
    combine = np.zeros((480,640,3))
    for i in range(len(NUMPYLIST) - 1):
#        print(i, 'called')
        
        kp1, desc1 = surf.detectAndCompute(NUMPYLIST[i], None)
        kp2, desc2 = surf.detectAndCompute(NUMPYLIST[i+1], None)
        
        i0 = cv2.drawKeypoints(NUMPYLIST[i], kp1, None, (255,0,0), 4)
        i1 = cv2.drawKeypoints(NUMPYLIST[i+1], kp2, None, (255,0,0), 4)
        
        xdelta, ydelta = SURF_FindMatches(kp1, kp2, desc1, desc2, i, i0, i1)
#        prevgray = cv2.cvtColor(NUMPYLIST[i], cv2.COLOR_BGR2GRAY)
#        gray = cv2.cvtColor(NUMPYLIST[i+1], cv2.COLOR_BGR2GRAY)
#        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
#        prevgray = gray
#    #            draw_flow(gray,flow)
#        bgr = draw_hsv(flow)
#        combine = np.add(combine, bgr)
        SURF_AddToHistograms(xdelta, ydelta)
        
    fvec = HISTOGRAM.getfeaturevec()
    HISTOGRAM.savehist(SUBJECTPATH)
    
    

#        print('Pair finished-----------------------')

        
#    print('DIFFERENCES STOPPED')

##-------------------------------------------------------------------------------





def Recognize_AD():
    infile = open(PATH + '/AD/trained-model.pkl','rb')
    clf = pickle.load(infile)
    imtest = np.array(Image.open(SUBJECTPATH + '\AD.jpeg').convert("L"))
    
    dcttest = dct( dct( imtest, axis=0), axis=1)
    
    featurevec2 = []
    for m in range(10):
        for n in range(10):
            featurevec2.append(dcttest[m][n])
    
    
    fvectest = np.array(featurevec2)
    
    fvectest = fvectest.reshape(1,-1)
    
#    y_pred = clf.predict_proba(fvectest)
#    print(y_pred)
    y_pred = clf.predict(fvectest)
    print(y_pred)
    return y_pred[0]



### ***************************************************************************
### ***************************************************************************
### MODIFY TO MAKE IT LIKE GEI
def Recognize_GEI():
    infile = open(PATH + '/GEI/trained-model.pkl','rb')
    clf = pickle.load(infile)
    imtest = np.array(Image.open(SUBJECTPATH + '/PADDED/result.jpeg').convert("L"))
    
    dcttest = dct( dct( imtest, axis=0), axis=1)
    
    featurevec2 = []
    for m in range(10):
        for n in range(10):
            featurevec2.append(dcttest[m][n])
    
    
    fvectest = np.array(featurevec2)
    
    fvectest = fvectest.reshape(1,-1)
    
#    y_pred = clf.predict_proba(fvectest)
#    print(y_pred)
    y_pred = clf.predict(fvectest)
    print(y_pred)
    return y_pred[0]
### GEI


### MODIFY TO MAKE IT LIKE SURF
def Recognize_SURF():
    infile = open(PATH + '/SURF/trained-model.pkl','rb')
    clf = pickle.load(infile)
#    imtest = np.array(Image.open(SUBJECTPATH + '\AD.jpeg').convert("L"))
    
    testvec = SURF_Histogram.loadfvec(SUBJECTPATH)
    testvec = np.array(testvec)
    testvec = testvec.reshape(1,-1)
    
    y_pred = clf.predict(testvec)
    print(y_pred)
    return y_pred[0]
### SURF
    



### MODIFY TO MAKE IT LIKE AMV
def Recognize_DOF():
    infile = open(PATH + '/DOF/trained-model.pkl','rb')
    clf = pickle.load(infile)
#    first = True
#    second = True
#    traininglabels = []
#    mainvec = []
#    firstvec = None
#    secondvec = None
#    
#    def DOF_fvecs():
#        for i in LABELS:                                          # open each walk for recorded person
#            walks = []                                              # array to store how many times they have walked
#            label_path = PATH_DOF + '/' + i                          # path to that person        
#            
#            
#            for walk in os.listdir(label_path):
#                walks.append(walk)                                  # get each walk
#                walk_path = label_path + '/' + walk                 # path to that walk
#                AD = DOF_Histogram.loadhist(walk_path)
#    #            print(AD)
#                dcttemp = dct( dct( AD, axis=0), axis=1)
#                AD = np.array(AD)
#    #            FVec = FVec.reshape(1,-1)
#                featurevec = []
#                
#                for m in range(10):
#                    for n in range(10):
#                        featurevec.append(dcttemp[m][n])
#                
#                FVec = np.array(featurevec)
#                FVec = FVec.reshape(1,-1)
#                
#                
#                global first, second, firstvec, secondvec, mainvec
#                if first and second:
#                    firstvec = FVec
#                    first = False
#                elif second:
#                    secondvec = FVec
#                    mainvec = np.append(firstvec, secondvec, axis = 0)
#                    second = False
#                else:
#                    mainvec = np.append(mainvec, FVec, axis = 0)
#                traininglabels.append(i)
#                    
#            print(i, walks)
#            
#    DOF_fvecs()
#    
##    tFinal = time()
##    print("Linear SVM training time = ", round(tFinal-tInit, 5), "s")
#    
#    
#    ##-----------------------for testing a single walk-----------------------------
#    
#    clf = svm.SVC(kernel='linear', C = 1.0)
#    clf.fit(mainvec, traininglabels)


    infile = open(SUBJECTPATH + '/AMV.pkl','rb')
    imtest = pickle.load(infile) 
    
    dcttest = dct( dct( imtest, axis=0), axis=1)
    
    featurevec2 = []
    for m in range(10):
        for n in range(10):
            featurevec2.append(dcttest[m][n])
    
    
    fvectest = np.array(featurevec2)
    
    fvectest = fvectest.reshape(1,-1)
    
#    y_pred = clf.predict_proba(fvectest)
#    print(y_pred)
    y_pred = clf.predict(fvectest)
    print(y_pred)
    return y_pred[0]
### AMV
### ***************************************************************************
### ***************************************************************************




################################################################################
## GET THE FEATURE VECTOR --> PRODUCE THE DCT IMAGE
################################################################################
def TerminateCapture():
    global KILLTHREAD
    KILLTHREAD = True
##-------------------------------------------------------------------------------











