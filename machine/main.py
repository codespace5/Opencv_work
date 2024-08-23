import cv2

img = cv2.imread('gray.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', gray)
cv2.waitKey(0)