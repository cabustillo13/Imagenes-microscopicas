# -*- coding: utf-8 -*-

""" Aplicar distintos algoritmos para determinar las esquinas de los elementos en una imagen""" 

import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('./Imagenes/granos/grains1.jpg')

# Path donde guardar los resultados
path = "./Imagenes/granos/resultados/"

###################
## HARRIS corner ##
###################

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray32 = np.float32(gray)  #Harris works on float32 images. 

# image, block size (size of neighborhood considered), ksize (aperture parameter for Sobel), k
harris = cv2.cornerHarris(gray32,2,3,0.04)  

# Threshold for an optimal value, it may vary depending on the image.
img[harris>0.01*harris.max()]=[255,0,0]    # replace these pixels with blue

cv2.imwrite(path + 'Harris_Corners.jpg',img)

#########################################################
## Shi-Tomasi Corner Detector & Good Features to Track ##
#########################################################

""" 
In opencv it is called goodfeaturestotrack
"""

#input image, #points, quality level (0-1), min euclidean dist. between detected points
corners = cv2.goodFeaturesToTrack(gray,50,0.05,10)
corners = np.int0(corners)   #np.int0 is int64

for i in corners:
    x,y = i.ravel()   # Ravel Returns a contiguous flattened array.
#    print(x,y)
    cv2.circle(img,(x,y),3,255,-1)  #Draws circle (Img, center, radius, color, etc.)

cv2.imwrite(path + 'Corners_goodfeaturestotrack.jpg',img)

#################
#SIFT and SURF ##
#################
"""
Don't work in OpenCV 3
SIFT significa (Stands for scale invariant feature transform)
"""

##########
## FAST ##
##########
"""
- FAST significa (Features from Accelerated Segment Test)
- High speed corner detector
- FAST is only keypoint detector. Cannot get any descriptors. 
"""

# Initiate FAST object with default values
detector = cv2.FastFeatureDetector_create(50)   #Detects 50 points

kp = detector.detect(img, None)

img2 = cv2.drawKeypoints(img, kp, None, flags=0)

cv2.imwrite(path + 'Corners_FAST.jpg',img2)

###########
## BRIEF ## 
###########

"""
- BRIEF significa (Binary Robust Independent Elementary Features)
- One important point is that BRIEF is a feature descriptor, 
- it doesn’t provide any method to find the features.
- Not going to show the example as BRIEF also not working in OpenCV 3
"""

#########
## ORB ##
#########
"""
- ORB significa (Oriented FAST and Rotated BRIEF)
- An efficient alternative to SIFT or SURF
- ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor
"""

orb = cv2.ORB_create(100)
kp, des = orb.detectAndCompute(img, None)

# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, None, flags=None)
# Now, let us draw with rich key points, reflecting descriptors. 
# Descriptors here show both the scale and the orientation of the keypoint.

img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(path + "With_keypoints.jpg", img2) 
