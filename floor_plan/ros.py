import cv2
import numpy as np
img = cv2.imread('1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ret, thresh_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)


# kernel = np.ones((3, 3), np.uint8)


# output = cv2.morphologyEx(gray, cv2.MORPH_CROSS, kernel)
ret, threah = cv2.threshold(gray, 10, 20, 100)
# contours, _ = cv2.findContours(threah, 1, 2)  
contours, _ = cv2.findContours(threah, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt  in contours:
    x1,y1 = cnt[0][0]
    if cv2.contourArea(cnt)> 8:
    # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    # print(approx)
        cv2.drawContours(img, [cnt], -1, (0,0,255), 30)
cv2.imwrite('out.png', img)
gray = cv2.resize(img, (1600, 1200))

cv2.imshow('gray', gray)

cv2.waitKey(0)