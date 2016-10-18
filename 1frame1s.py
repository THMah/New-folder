import cv2
import math

videoFile = "770_5.mp4"
imagesFolder = "/home/th/Desktop/bus_dataset_2/770"
cap = cv2.VideoCapture(videoFile)
#frame rate
frameRate = cap.get(5)
while(cap.isOpened()):
#current frame number
    frameId = cap.get(1)
    ret, frame = cap.read()
    flipFrame = cv2.flip(frame, 1)
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = imagesFolder + "/770_5_" +  str(int(frameId)) + ".jpg"
        filename2 = imagesFolder + "/flip_770_5_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
        cv2.imwrite(filename2, flipFrame)
cap.release()
print "Done!"