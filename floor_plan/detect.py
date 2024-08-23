import cv2
import numpy as np

img = cv2.imread('1.jpg')
# lower = np.array([50, 80, 80], dtype='uint8')
# upper = np.array([255, 130, 130], dtype='uint8')
lower = np.array([130, 130, 220], dtype=np.uint8)
upper = np.array([160, 160, 255], dtype=np.uint8)

mask = cv2.inRange(img, lower, upper)

detected_color = cv2.bitwise_and(img, img, mask=mask)


detected_gray = cv2.cvtColor(detected_color, cv2.COLOR_BGR2GRAY)

cv2.imshow('detect', detected_color)
# contours, _ = cv2.findContours(detected_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# centers  = []

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.circle(img=img, center=(int(x+ h/2), int(y+ w/2)), radius=5, thickness=2, color=(255, 0, 0))


# img = cv2.resize(img, (1600, 1200))
# cv2.imshow("img", img)
cv2.waitKey(0)