import cv2
import numpy as np
img = cv2.imread('img/balls-1-03.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', gray)
num = 0
img_blue = cv2.GaussianBlur(gray, (3,3), 0)

cv2.imshow('idd', img_blue)


edge = cv2.Canny(image=img_blue, threshold1=100, threshold2=200)

cv2.imshow('img', edge)


detected_circle = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=20,minRadius=0,maxRadius=16 )

if detected_circle is not None:
    for pt in detected_circle[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        if r > 2 and r < 16.1:
            print('radus', r, a, b)
            img2 = img.copy()
            cv2.imshow('111', img)

            cv2.circle(img2, (int(a), int(b)), int(r), (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            # cv2.circle(img2, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img2)
            # cv2.imshow('img', gray_blurred)
            cv2.waitKey(0)




# edge = cv2.Canny(img = gray, threshold1=100, threshold2=200)
# detected_circles = cv2.HoughCircles(edge,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=50,maxRadius=300)
# detected_circles = cv2.HoughCircles(detected_circles,cv2.HOUGH_GRADIENT,1,20,
#                         param1=50,param2=30,minRadius=50,maxRadius=300)

# if detected_circles is not None:

#         # Convert the circle parameters a, b and r to integers.
#         detected_circles = np.uint16(np.around(detected_circles))

#         for pt in detected_circles[0, :]:
#             num += 1
#             a, b, r = pt[0], pt[1], pt[2]
#             if r>=0.1:
#             # Draw the circumference of the circle.
#                 print('radus', r)
#                 img2 = img
#                 cv2.imshow('111', img)

#                 cv2.circle(img2, (a, b), r, (0, 255, 0), 2)
        
#                 # Draw a small circle (of radius 1) to show the center.
#                 cv2.circle(img2, (a, b), 1, (0, 0, 255), 3)
#                 cv2.imshow("Detected Circle", img2)
#                 # cv2.imshow('img', gray_blurred)
#                 cv2.waitKey(0)
#             break

cv2.waitKey(0)