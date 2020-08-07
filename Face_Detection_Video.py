import cv2
from random import randrange
# randrange picks a color upto a given range excluding that range like if randrange(256) => will give number between 0-255

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# To capture video from webcam, videoCapture(0) the 0 automatically goes to default webcam
webcam = cv2.VideoCapture(0)

# To use a video from laptop instead of webcam
webcam = cv2.VideoCapture('VID.mp4')

# Iterates forever till the video is ended or a key is pressed
while True:
    # Read the current frame and get the image out of the video on a LOOP
    #frame is the image we get from the webcam
    successful_frame_read, frame = webcam.read()

    # Must convert the frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and coordinates of rectangles [[x-axis y-axis width height]] eg: [[140 182 577 577]] where first 2 values are x and y distance from top and topLeft to which next 2 coordinates are added to form rectangle
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draws rectangle around the faces in frame
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256),
                                                  randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector', frame)

    # This key is automatically pressed after a delay of 1sec for the video frame to change
    #if we dont have this waitkey the video wont play
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the videoCapture object to perform cleanup
webcam.release()


"""
# Detect faces and coordinates of rectangles [[x-axis y-axis width height]] eg: [[140 182 577 577]] where first 2 values are x and y distance from top and topLeft to which next 2 coordinates are added to form rectangle
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# Draws rectangle around the faces with [[140 182 577 577]] coordinates and BGR color (here green) and a thickness of 2
# cv2.rectangle(img, (104, 182), (104+577, 182+577), (0, 255, 0), 2)
# (x, y, w, h) = face_coordinates[0]
# for (x, y, w, h) in range (len(face_coordinates)):
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),
                                            randrange(256), randrange(256)), 2)
print(face_coordinates)
# cv2 function to display the read image faces
cv2.imshow('Face Detector', img)
# This keeps the window paused until a key is pressed
cv2.waitKey()
"""

print("Code Completed")