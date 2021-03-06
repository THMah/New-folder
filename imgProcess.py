import numpy as np
import cv2
import os
import pytesseract
from PIL import Image
import math
import datetime

IMAGE_FILE = "/home/th/Desktop/FYP_latest/BusDetector/frame10.jpg"
OCR_VIDEO_FRAME = '/home/th/Desktop/FYP_latest/BusDetector/OCR_video_frame'


#use 0 to capture live video / VideoPath to play local video
cap = cv2.VideoCapture("/home/th/Desktop/FYP_latest/BusDetector/bus video oct/Editted/bus_771_5.mp4")
#frame rate
frameRate = cap.get(5)
while(cap.isOpened()):
#current frame number
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        now = format(datetime.datetime.now())
        filename = OCR_VIDEO_FRAME + "/" + str(now) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
#end capture video and save as image

for file in sorted(os.listdir(OCR_VIDEO_FRAME)):
    if file.endswith(".jpg"):
        print(file)

        img = cv2.imread(OCR_VIDEO_FRAME + '/' + file)
        #img = cv2.imread(IMAGE_FILE)

        #convert image into hsv
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #set min and max color range in hsv
        #yellow
        #COLOR_MIN = np.array([0, 100, 100], np.uint8)
        #COLOR_MAX = np.array([40, 255, 255], np.uint8)
        #red hsv_red = 0,100,50
        COLOR_MIN = np.array([0, 100, 50], np.uint8)
        COLOR_MAX = np.array([10, 255, 255], np.uint8)

        #locate and set threshold
        frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
        imgray = frame_threshed
        ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        x, y, w, h = cv2.boundingRect(cnt)
        rectangle = cv2.rectangle(img, (x, y), (x + w, y + h),
            (0, 255, 0), 2)
        cv2.imshow("Show", img)
        busLine = img[y: y + h, x: x + w]

        #zooming in the image, better viewing
        resizeBusLine = cv2.resize(busLine, None, fx=5, fy=5,
             interpolation=cv2.INTER_LINEAR)
        cv2.imshow("busline", resizeBusLine)
        grayimg = cv2.cvtColor(resizeBusLine, cv2.COLOR_BGR2GRAY)

        im_bw = cv2.adaptiveThreshold(grayimg, 255, cv2.THRESH_BINARY,
        cv2.THRESH_BINARY, 11, 2)
        cv2.imshow("bw", im_bw)
        kernel = np.ones((3, 2), np.int8)
        opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(im_bw, kernel, iterations=3)
        dilation = cv2.dilate(im_bw, kernel, iterations=1)

        ret, bw1 = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, bw2 = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, bw3 = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imshow("erosion", bw1)
        cv2.imshow("opening", bw2)
        cv2.imshow("dilation", bw3)
        edges = cv2.Canny(bw2, 100,200)
        cv2.imwrite("character.jpg", edges)
        cv2.imshow("edges",edges)
        cv2.waitKey()
        cv2.destroyAllWindows()

