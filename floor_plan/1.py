import cv2
import numpy as np

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1,1), np.uint8)
# erosion = cv2.dilate(gray, kernel, iterations=1)
# closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# erosion = cv2.resize(closing, (2000, 1600))
# cv2.imshow("eros", erosion)
# cv2.waitKey(0)

ret,thresh = cv2.threshold(gray,25,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
print("Number of contours detected:", len(contours))

for cnt in contours:
   x1,y1 = cnt[0][0]
   approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      ratio = float(w)/h
      if (ratio >= 2 and ratio <= 2.2) or ((ratio >= 2.8 and ratio <= 3.2)):
         img = cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
         cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      # if:
      #    cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      #    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
img = cv2.resize(img, (2000, 1600))
cv2.imshow("Shapes", img)

print('3444444')
cv2.waitKey(0)
cv2.destroyAllWindows()