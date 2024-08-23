import numpy as np
import cv2
img = cv2.imread('1.jpg')
img = cv2.resize(img, (800, 600))

lower  = np.array([52, 100, 76], dtype='uint8')
higher = np.array([160, 175, 175], dtype='uint8')
mask = cv2.inRange(img, lower, higher)
detected_img = cv2.bitwise_and(img, img, mask= mask)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blue = cv2.GaussianBlur(gray, (3, 3), 0)
# edge = cv2.Canny(image=img, threshold1=120, threshold2=180)
# # t, edge = cv2.threshold(blue, 130, 255, cv2.THRESH_BINARY)
# cv2.imshow('edg', edge)

# detected_circle = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5,
#                                    maxRadius= 100)
# print(detected_circle)
# if detected_circle is not None:
#     detected_circle = np.uint16(np.around(detected_circle))
#     for pt in detected_circle[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#         if r> 5:
#             cv2.circle(img, (a,b), r, (0, 255, 0), 2)
#             cv2.circle(img, (a,b), 1, (0, 255, 0), 3)
#             cv2.imshow('img', img)
#             cv2.waitKey(0)

# cv2.imshow('gray', edge)
cv2.imshow('ig',detected_img)
cv2.waitKey(0)