import cv2
import numpy as np

img = cv2.imread('1.png')
img = img[450:-30, :]
cv2.imshow("image", img)

lower = np.array([220, 220, 220], dtype='uint8')
upper = np.array([255, 255, 255], dtype='uint8')
mask = cv2.inRange(img, lower, upper)
detected_color = cv2.bitwise_and(img, img, mask=mask)

# Convert detected_color to grayscale before finding contours
detected_gray = cv2.cvtColor(detected_color, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(detected_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
centers = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.circle(img=img, center=(int(x+ h/2), int(y+ w/2)), radius=5, thickness=2, color=(255, 0, 0))

cv2.imshow('color', detected_color)
cv2.imshow('contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()