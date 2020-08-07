import cv2
from random import randrange
# //this is used to detect the front facing of all the people
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img=cv2.imread('RDJ.jpg')

# Must convert the img to grayscale
#convert that to gray and we write opposite that is bgr rather than rgb
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces and coordinates of rectangles [[x-axis y-axis width height]] eg: [[140 182 577 577]] where first 2 values are x and y distance from top and topLeft to which next 2 coordinates are added to form rectangle
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Draws rectangle around the faces with [[140 182 577 577]] coordinates and BGR color (here green) and a thickness of 2
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(128, 256),randrange(256), randrange(256)), 2)
# 

#this will show the image which is feded in the img so it is opencv function to show the img
cv2.imshow('Face Detector',img)

#this will provide a wait to the pic once it is being imported and we can press any key to continue
cv2.waitKey()

print("code complete")
