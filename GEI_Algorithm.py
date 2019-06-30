# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:18:15 2019

@author: Asif Towheed
"""

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################


import time
from PIL import Image
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from imutils.object_detection import non_max_suppression
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
l = 0

PATH = os.getcwd().replace('\\','/')

def detect(gray, frame, grey_color, depth_image_3d, clipping_distance): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    rects, weights = hog.detectMultiScale(gray)
    for i, (x, y, w, h) in enumerate(rects): # For each detected face:
        print("print: ", weights[i] < 0.7)
        if weights[i] < 0.7:
            print("reached continue")
            continue
        print("passed continue")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    pick = non_max_suppression(rects, probs=None,overlapThresh=0.65)

    for(xA,yA,xB,yB) in pick:
        cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0),2)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, frame)
        person = bg_removed[y:y+h, x:x+w]

        pilimg = Image.fromarray(person)
        print("saving")
        global l
        l = l + 1
        pilimg.save('person-{}.jpeg'.format(l))

        return frame # We return the image with the detector rectangles.

def detect2(imagearr, depth, clipdist):
    """
    print (len(imagearr))   # 480 = number of rows (column-wise)
    for x in imagearr:      # row-wise interation
        print(x)            # each pixel in a row (column-wise)
        print(x.size)
        #print(y)
    """
    imagearr

    for y in range(480):
        for x in range(640):
            dist = depth.get_distance(x,y)
            
            if 0 < dist and dist < clipdist:
                imagearr[y][x] = 255

    return imagearr

    


def finddiff(im1, im2):
    
    sumofdiff = 0
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    height = h1
    width = w1
    individualdiff2 = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
    
    print("h1",h1,"w1",w1," h2",h2,"w2",w2)
    
    for w in range(width):
        for h in range(height):
            absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
            #print("PRINTING", int(im1[h,w]))
            if absdiff < 30:
                absdiff = 0
            sumofdiff += absdiff
            individualdiff2[h,w] = absdiff
    
    newthresh = sumofdiff/(width*height)
    
    for w in range(width):
        for h in range(height):
            if individualdiff2[h,w] < newthresh:
                individualdiff2[h,w] = 0
#            else:
#                accumdiff2[h,w] += 10
    
            
    newimage = Image.fromarray(individualdiff2)
    return newimage



def mainrun():
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
            depthwhite = detect2(gray, aligned_depth_frame, clipping_distance*depth_scale) # We get the output of our detect function.
    
            # Render images
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', depthwhite)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"): # Exit condition
                cv2.destroyAllWindows()
                sumw = sumh = maxw = maxh = 0
                count = 0
                global PATH
                for filepath in glob.iglob(PATH + '/*.jpeg'):
                    path = PATH + '/*.jpeg'
                    print(filepath)
                    print(path)
                    ori = cv2.imread(filepath)
                    height, width, depth = ori.shape
                    sumw += width
                    sumh += height
                    if width > maxw:
                        maxw = width
                    if height > maxh:
                        maxh = height
                    print(width, height)
                    count += 1
    
                finalw = sumw/count
                finalh = sumh/count
                print('count', count)
                result = None
                checker = True
    
                npimages = []
    
    
                for filepath in glob.iglob(PATH + '/*.jpeg'):
                    ori = cv2.imread(filepath)
                    height, width, depth = ori.shape
                    print(width, height)
                    if width < finalw:
                        print('removing', filepath)
                        os.unlink(filepath)
                    else:
                        #newimg = cv2.resize(ori,(int(maxw),int(maxh)))
                        to_be_added = (maxw-width)
                        padval = math.floor(to_be_added/2)
                        print("PADVAL ", padval)
                        if (to_be_added%2) == 0:
                            print("ENTERED!!!!!!!!!!!!!!")
                            newimg = np.pad(ori, ((0,0),(padval,padval), (0,0)), mode="constant")
                        else:
                            print("22222ENTERED!!!!!!!!!!!!!!")
                            newimg = np.pad(ori, ((0,0),(padval,padval+1), (0,0)), mode="constant")
                        newimg = cv2.resize(newimg,(int(maxw),int(maxh)))
                        if checker is True:
                            print(checker)
                            print("ENTERED!!!!!!!!!!!!")
                            result = Image.fromarray(newimg)
                            checker = False
                            print(checker)
                        pilimg = Image.fromarray(newimg)
                        pilimg.save(filepath)
                        npimages.append(np.array(Image.open(filepath).convert("L")))
    
                for i in range(len(npimages) - 1):
                    #newimg = finddiff(npimages[i], npimages[i+1])
                    #h1, w1 = np.array(result).size
                    #h2, w2 = newimg.size
                    #print("h1",h1,"w1",w1," h2",h2,"w2",w2)
                    print(result == None)
                    pilimg = Image.fromarray(np.array(result))
                    pilimg.save(PATH + '/result.jpeg')
                    pilimg = Image.fromarray(np.array(npimages[i]))
                    pilimg.save(PATH + '/{}.jpeg'.format(i))
                    newimg = cv2.imread(PATH + '/{}.jpeg'.format(i))
                    result = cv2.imread(PATH + '/result.jpeg')
                    
                    result = cv2.addWeighted(result, 0.5, newimg, 0.5, 0)    
    
                pilimg2 = Image.fromarray(np.array(result))
                pilimg2.save('GEI.jpeg')
                break
    finally:
        pipeline.stop()
    
    








# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Fri Apr  5 14:18:15 2019
# 
# @author: Asif Towheed
# """
# 
# ## License: Apache 2.0. See LICENSE file in root directory.
# ## Copyright(c) 2017 Intel Corporation. All Rights Reserved.
# 
# #####################################################
# ##              Align Depth to Color               ##
# #####################################################
# 
# 
# import time
# from PIL import Image
# # First import the library
# import pyrealsense2 as rs
# # Import Numpy for easy array manipulation
# import numpy as np
# # Import OpenCV for easy image rendering
# import cv2
# from imutils.object_detection import non_max_suppression
# import glob
# import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import math
# 
# 
# 
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# l = 0
# 
# PATH = os.getcwd().replace('\\','/')
# 
# def detect(gray, frame, grey_color, depth_image_3d, clipping_distance): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
#     rects, weights = hog.detectMultiScale(gray)
#     for i, (x, y, w, h) in enumerate(rects): # For each detected face:
#         print("print: ", weights[i] < 0.7)
#         if weights[i] < 0.7:
#             print("reached continue")
#             continue
#         print("passed continue")
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
#     rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
#     pick = non_max_suppression(rects, probs=None,overlapThresh=0.65)
# 
#     for(xA,yA,xB,yB) in pick:
#         cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0),2)
#         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
#         
#         bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, frame)
#         person = bg_removed[y:y+h, x:x+w]
# 
#         pilimg = Image.fromarray(person)
#         print("saving")
#         global l
#         l = l + 1
#         pilimg.save('person-{}.jpeg'.format(l))
# 
#         return frame # We return the image with the detector rectangles.
# 
# def detect2(imagearr, depth, clipdist):
#     """
#     print (len(imagearr))   # 480 = number of rows (column-wise)
#     for x in imagearr:      # row-wise interation
#         print(x)            # each pixel in a row (column-wise)
#         print(x.size)
#         #print(y)
#     """
#     lowest = rightmost = 1
#     topmost = 480
#     leftmost = 640
#     detected = False
#     imagearr2 = np.copy(imagearr)
#     
#     
# #    xpos = 0
# #    for x in imagearr:
# #        ypos = 0
# #        for y in x:
# #            if y > 75:      #non-bg pixel found
# #                #print(y)
# #                if y < 200:
# #                    x[ypos] = 255
# #                else:
# #                    x[ypos] = 0
# #                if xpos < topmost:
# #                    topmost = xpos
# #                    detected = True
# #                if xpos > lowest:
# #                    lowest = xpos
# #                    detected = True
# #                if ypos > rightmost:
# #                    rightmost = ypos
# #                    detected = True
# #                if ypos < leftmost:
# #                    leftmost = ypos
# #                    #print(y)
# #                    #print(ypos,"\n")
# #                    detected = True
# #
# #            ypos += 1
#             #print(y)
# #        xpos += 1
# 
#     for y in range(480):
#         for x in range(640):
#             dist = depth.get_distance(x,y)
#             
#             if 0 < dist and dist < clipdist:
#                 #print(dist, clipdist)
#                 #imagearr2[y][x] += 50
#                 #if imagearr2[y][x] > 255:
#                 imagearr2[y][x] = 255
#                 if y < topmost:
#                     topmost = y
#                     detected = True
#                 if y > lowest:
#                     lowest = y
#                     detected = True
#                 if x > rightmost:
#                     rightmost = x
#                     detected = True
#                 if x < leftmost:
#                     leftmost = x
#                     #print(y)
#                     #print(ypos,"\n")
#                     detected = True
# 
# 
# 
# 
# 
# #    print("DONE\n\n\nDONE\n\n\nDONE\n\n\nDONE\n\n\nDONE\n\n\n")
# #    print("top", topmost)
# #    print("lowest", lowest)
# #    print("left", leftmost)
# #    print("right", rightmost)
#     if detected is True and leftmost != rightmost and topmost != lowest:
#         bbox = (leftmost, topmost, rightmost, lowest)
#         print(bbox)
#         #person = imagearr[topmost:lowest-topmost, leftmost:rightmost-leftmost]
#         pilimg = Image.fromarray(imagearr2).crop(bbox)
# #        print("saving")
#         global l
#         l = l + 1
# #        pilimg.show()
#         pilimg.save('person-{}.jpeg'.format(l))
#     return imagearr, imagearr2
# 
#     
# 
# 
# def finddiff(im1, im2):
#     
#     sumofdiff = 0
#     h1, w1 = im1.shape
#     h2, w2 = im2.shape
#     height = h1
#     width = w1
#     individualdiff2 = np.zeros([height, width], dtype=np.uint8) # to store the individual differences
#     
#     print("h1",h1,"w1",w1," h2",h2,"w2",w2)
#     
#     for w in range(width):
#         for h in range(height):
#             absdiff = abs(int(im1[h,w]) - int(im2[h,w]))
#             #print("PRINTING", int(im1[h,w]))
#             if absdiff < 30:
#                 absdiff = 0
#             sumofdiff += absdiff
#             individualdiff2[h,w] = absdiff
#     
#     newthresh = sumofdiff/(width*height)
#     
#     for w in range(width):
#         for h in range(height):
#             if individualdiff2[h,w] < newthresh:
#                 individualdiff2[h,w] = 0
# #            else:
# #                accumdiff2[h,w] += 10
#     
#             
#     newimage = Image.fromarray(individualdiff2)
#     return newimage
# 
# 
# 
# def mainrun():
#     #################################################################################
#     # Create a pipeline
#     pipeline = rs.pipeline()
#     
#     #Create a config and configure the pipeline to stream
#     #  different resolutions of color and depth streams
#     config = rs.config()
#     config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     
#     # Start streaming
#     profile = pipeline.start(config)
#     #################################################################################
#     
#     
#     # Getting the depth sensor's depth scale (see rs-align example for explanation)
#     depth_sensor = profile.get_device().first_depth_sensor()
#     depth_scale = depth_sensor.get_depth_scale()
#     print("Depth Scale is: " , depth_scale)
#     
#     # We will be removing the background of objects more than
#     #  clipping_distance_in_meters meters away
#     clipping_distance_in_meters = 3 #5.5 meter optimum for gait recognition
#     clipping_distance = clipping_distance_in_meters / depth_scale
#     
#     # Create an align object
#     # rs.align allows us to perform alignment of depth frames to others frames
#     # The "align_to" is the stream type to which we plan to align depth frames.
#     align_to = rs.stream.color
#     align = rs.align(align_to)
#     
#     # Streaming loop
#     try:
#         while True:
#             # Get frameset of color and depth
#             frames = pipeline.wait_for_frames()
#             # frames.get_depth_frame() is a 640x360 depth image
#             
#             # Align the depth frame to color frame
#             aligned_frames = align.process(frames)
#             
#             # Get aligned frames
#             aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#             color_frame = aligned_frames.get_color_frame()
#             
#             # Validate that both frames are valid
#             if not aligned_depth_frame or not color_frame:
#                 continue
#             
#             depth_image = np.asanyarray(aligned_depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())
#             
#             # Remove background - Set pixels further than clipping_distance to grey
#             grey_color = 0
#             depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
#             bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
#             
#             gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
#             # detect (using HOG)
#             #canvas = detect(gray, color_image, grey_color, depth_image_3d, clipping_distance) # We get the output of our detect function.
#             # detect (manual) ################################################################################################################################################################################
#             bg_removed2, depthwhite = detect2(gray, aligned_depth_frame, clipping_distance*depth_scale) # We get the output of our detect function.
#     
#             # Render images
#             #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#             #images = np.hstack((bg_removed, depth_colormap))
#             cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('Align Example', depthwhite)
#             k = cv2.waitKey(1)
#             if k & 0xFF == ord("q"): # Exit condition
#                 cv2.destroyAllWindows()
#                 sumw = sumh = maxw = maxh = 0
#                 count = 0
#                 global PATH
#                 for filepath in glob.iglob(PATH + '/*.jpeg'):
#                     path = PATH + '/*.jpeg'
#                     print(filepath)
#                     print(path)
#                     ori = cv2.imread(filepath)
#                     height, width, depth = ori.shape
#                     sumw += width
#                     sumh += height
#                     if width > maxw:
#                         maxw = width
#                     if height > maxh:
#                         maxh = height
#                     print(width, height)
#                     count += 1
#     
#                 finalw = sumw/count
#                 finalh = sumh/count
#                 print('count', count)
#                 result = None
#                 checker = True
#     
#                 npimages = []
#     
#     
#                 for filepath in glob.iglob(PATH + '/*.jpeg'):
#                     ori = cv2.imread(filepath)
#                     height, width, depth = ori.shape
#                     print(width, height)
#                     if width < finalw:
#                         print('removing', filepath)
#                         os.unlink(filepath)
#                     else:
#                         #newimg = cv2.resize(ori,(int(maxw),int(maxh)))
#                         to_be_added = (maxw-width)
#                         padval = math.floor(to_be_added/2)
#                         print("PADVAL ", padval)
#                         if (to_be_added%2) == 0:
#                             print("ENTERED!!!!!!!!!!!!!!")
#                             newimg = np.pad(ori, ((0,0),(padval,padval), (0,0)), mode="constant")
#                         else:
#                             print("22222ENTERED!!!!!!!!!!!!!!")
#                             newimg = np.pad(ori, ((0,0),(padval,padval+1), (0,0)), mode="constant")
#                         newimg = cv2.resize(newimg,(int(maxw),int(maxh)))
#                         if checker is True:
#                             print(checker)
#                             print("ENTERED!!!!!!!!!!!!")
#                             result = Image.fromarray(newimg)
#                             checker = False
#                             print(checker)
#                         pilimg = Image.fromarray(newimg)
#                         pilimg.save(filepath)
#                         npimages.append(np.array(Image.open(filepath).convert("L")))
#     
#                 for i in range(len(npimages) - 1):
#                     #newimg = finddiff(npimages[i], npimages[i+1])
#                     #h1, w1 = np.array(result).size
#                     #h2, w2 = newimg.size
#                     #print("h1",h1,"w1",w1," h2",h2,"w2",w2)
#                     print(result == None)
#                     pilimg = Image.fromarray(np.array(result))
#                     pilimg.save(PATH + '/result.jpeg')
#                     pilimg = Image.fromarray(np.array(npimages[i]))
#                     pilimg.save(PATH + '/{}.jpeg'.format(i))
#                     newimg = cv2.imread(PATH + '/{}.jpeg'.format(i))
#                     result = cv2.imread(PATH + '/result.jpeg')
#                     
#                     result = cv2.addWeighted(result, 0.5, newimg, 0.5, 0)    
#     
#                 pilimg2 = Image.fromarray(np.array(result))
#                 pilimg2.save('GEI.jpeg')
#                 break
#     finally:
#         pipeline.stop()
#     
#     
# 
# 
# 
# 
# 
# 
# =============================================================================
