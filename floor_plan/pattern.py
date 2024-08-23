import cv2
import numpy as np

img = cv2.imread('1.jpg')

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_purple = np.array([120, 50, 50])
upper_purple = np.array([160, 255, 255])

purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

result = cv2.bitwise_and(img, img, mask=purple_mask)

result = cv2.bitwise_not(result)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

kernel = np.ones((9,9), np.uint8)
ret,thresh = cv2.threshold(gray,200,255,0)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)

thes1 = cv2.resize(thresh, (800, 800))
cv2.imshow('thes', thes1)

# contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

# print(contours)
img_name = "pattern"
k = 0
for cnt in contours:
    k +=1
    box = cv2.boundingRect(cnt)
    print(box)

    if(box[2] < 500):

      pt1 = [box[0], box[1]]
      pt2 = [box[0] + box[2] , box[1] + box[3] ]

      cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=[0, 255, 0], thickness=12)
    #   img_pattern = img[box[0]:box[0] + box[2] +500, box[1]:box[1] + box[3]+ 500]
    #   img_pattern = img[box[1] + box[3]:box[1] + box[3]+ 80, box[0] + box[2]:box[0] + box[2]+ 80]
      pattern = img[box[1] -30:box[1]  + box[3] + 50, box[0]-30:box[0] + box[2]] + 30
      img_pattern = img[box[1] + box[3] + 10:box[1] + box[3]+ 30, box[0]:box[0]+ 30]
      cv2.imshow('mark', pattern)
      img_pattern = cv2.resize(img_pattern, (120, 120))
      cv2.imshow('patter', img_pattern)

      cv2.waitKey(0)
#    x1,y1 = cnt[0][0]

#    img = cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
#    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#    if len(approx) == 4:
#       x, y, w, h = cv2.boundingRect(cnt)
#       img = cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
#       cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

img = cv2.resize(img, (1000, 1000))
res = cv2.resize(gray, (1000, 1000))
cv2.imshow("Shapes", img)

# cv2.imshow('Result', res)
cv2.waitKey(0)
# cv2.destroyAllWindows()