import numpy as np
import cv2

img = cv2.imread('frame10.jpg')
#remove bottom part of image
crop_img = img[0:250, 0:1000]

#convert image into hsv
hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

#set min and max color range in hsv
COLOR_MIN = np.array([20, 80, 80], np.uint8)
COLOR_MAX = np.array([40, 255, 255], np.uint8)

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
rectangle = cv2.rectangle(crop_img, (x - 70, y - 5), (x + w + 20, y + h + 5),
    (0, 255, 0), 2)
cv2.imshow("Show", crop_img)
#hard coded value to crop the bus line region
busLine = crop_img[y: y + h + 5, x - 70: x + w + 10]

#zooming in the image, better viewing
resizeBusLine = cv2.resize(busLine, None, fx=2, fy=2,
     interpolation=cv2.INTER_LINEAR)

kernel = np.ones((1, 1), np.int8)
#opening = cv2.morphologyEx(resizeBusLine, cv2.MORPH_OPEN, kernel)
#erosion = cv2.erode(resizeBusLine,kernel,iterations = 3)
#dilation = cv2.dilate(busLine, kernel, iterations=1)
#cv2.imshow("Detected Bus Line", erosion)
cv2.waitKey()
cv2.destroyAllWindows()