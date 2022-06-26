import numpy as np # linear algebra
import cv2
from matplotlib import pyplot as plt

#----------------This code not ready yet---------------
img = cv2.imread('data-for-knn/data/train/airplane/0000.jpg')
plt.imshow(img)
plt.show()
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#create sift detector
sift = cv2.SIFT_create()
#finds th keypoints in images
kp = sift.detect(gray,None)

#this function draws the small circules on location of keypoints
img =  cv2.drawKeypoints(gray,kp,img)
plt.imshow(img)
plt.show()

#draw a circle with size of keypoint and it will even show its orientation
img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img)
plt.show()

#-------------------create our descripteur--------------------
#mmethode 1: if we already have key-point kp
desc = sift.compute(gray,kp)
print(desc)
#methode 2: detect and compute in sametime
kp, des = sift.detectAndCompute(gray,None)
des
