import numpy as np
import cv2

img = cv2.imread('frame10.jpg')
#remove bottom part of image
crop_img = img[0:250, 0:1000]

#convert image into hsv
hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

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
#rectangle = cv2.rectangle(crop_img, (x - 70, y - 5), (x + w + 20, y + h + 5),
#    (0, 255, 0), 2)
rectangle = cv2.rectangle(crop_img, (x , y ), (x + w , y + h),
    (0, 255, 0), 2)
cv2.imshow("Show", crop_img)
#hard coded value to crop the bus line region
#busLine = crop_img[y: y + h + 5, x - 70: x + w + 10]
busLine = crop_img[y: y + h, x: x + w]

#zooming in the image, better viewing
resizeBusLine = cv2.resize(busLine, None, fx=5, fy=5,
     interpolation=cv2.INTER_LINEAR)

#2D filter
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#resizeBusLine = cv2.filter2D(resizeBusLine, -1, kernel)


kernel = np.ones((1, 1), np.int8)
opening = cv2.morphologyEx(resizeBusLine, cv2.MORPH_OPEN, kernel)
#erosion = cv2.erode(resizeBusLine,kernel,iterations = 3)
#dilation = cv2.dilate(busLine, kernel, iterations=1)
grayimg = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
#thresh = 127
#im_bw = cv2.threshold(grayimg, thresh, 255, cv2.THRESH_BINARY)[1]
im_bw = cv2.adaptiveThreshold(grayimg,255,cv2.THRESH_BINARY,\
cv2.THRESH_BINARY,11,2)

#erosion = cv2.erode(resizeBusLine,kernel,iterations = 3)
#dilation = cv2.dilate(busLine, kernel, iterations=1)
cv2.imshow("Detected Bus Line", im_bw)
cv2.waitKey()
cv2.destroyAllWindows()