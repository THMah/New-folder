import numpy as np
import cv2

#1st argument is to specific name
#2nd argument is specify how does image is being read
#1 = color img, 0 = grayscale, -1 = load as usual
busImg = cv2.imread('frame10.jpg', 1)

#define boundaries
boundaries = [([17, 15, 100], [50, 56, 200]),
    ([25, 146, 190], [62, 174, 250])]
for (lower, upper) in boundaries:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')
        mask = cv2.inRange(busImg, lower, upper)
        output = cv2.bitwise_and(busImg, busImg, mask=mask)

cv2.imshow('bus image', np.hstack([busImg, output]))
cv2.waitKey(0)
#cv2.DestroyAllWindows()