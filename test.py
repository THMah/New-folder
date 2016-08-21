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
cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Show", crop_img)
#cv2.imshow("Ori", img)
cv2.waitKey()
cv2.destroyAllWindows()