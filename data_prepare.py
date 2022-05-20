from time import sleep
import cv2
import os

for dir in os.listdir('data/test'):
    for f in dir:
       pass

image = cv2.imread('data/test/0000.jpg')
cv2.imshow(image)
window_name = 'image'
  
# Using cv2.imshow() method 
# Displaying the image 
cv2.imshow('', image)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 