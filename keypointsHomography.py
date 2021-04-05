# -*- coding: utf-8 -*-

"""Keypoints homography for registration in OpenCV """

###############################################
## Brute-Force Matching with ORB Descriptors ##
###############################################

"""
Brute-Force matcher takes the descriptor of one feature in first set and is 
matched with all other features in second set using some distance calculation.
create Matcher object
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Cargar imagenes
im1 = cv2.imread('./Imagenes/match/monkey_distorted.jpg')          # Image that needs to be registered.
im2 = cv2.imread('./Imagenes/match/monkey.jpg') # trainImage

#Path donde se guardaran los resultados
path = "./Imagenes/match/"

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(50)  #Registration works with at least 50 points

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#Like we used cv2.drawKeypoints() to draw keypoints, 
#cv2.drawMatches() helps us to draw the matches. 
#https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imwrite(path + "Matches_image.jpg", img3)

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

#Now we have all good keypoints so we are ready for homography.     
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
# Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))  #Applies a perspective transformation to an image.
   
print("Estimated homography : \n",  h)

cv2.imwrite(path + "Registered_image.jpg", im1Reg)
