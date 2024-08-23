import cv2
import numpy as np

def _main():
    img = cv2.imread('Capture1.JPG')
    result_image = img.copy()
    (h, w, _) = img.shape

    lowers = [np.array([220, 210, 220]), np.array([182, 145, 140]), np.array([152, 140, 148])]
    uppers = [np.array([250, 255, 250]), np.array([196, 160, 155]), np.array([172, 162, 168])]

    roof_range = np.zeros((h, w), np.uint8)

    for i in range(len(lowers)):
        lower = lowers[i]
        upper = uppers[i]
        tmp_roof_range = cv2.inRange(img, lower, upper)
        roof_range = cv2.bitwise_or(roof_range, tmp_roof_range)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,50,150,apertureSize = 3)
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(roof_range, cv2.MORPH_OPEN, kernel=kernel)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=kernel)

    cv2.imshow("roof_range", roof_range)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(closing, 0, 1)

    # black_image = np.zeros((h, w), np.uint8)
    real_contours = []
    for contour in contours:
        black_image = np.zeros((h, w), np.uint8)
        cv2.drawContours(black_image, [contour], 0, 255 ,-1)
        if np.count_nonzero(black_image) / (h * w) > 0.001:
            real_contours.append(contour)

    # cv2.imshow("masked", black_image)
    # cv2.waitKey(0)

    black_image = np.zeros((h, w), np.uint8)
    for contour in real_contours:
        cv2.drawContours(black_image, [contour], 0, 255 ,-1)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
        box = np.int0(box)
        cv2.drawContours(result_image,[box],0,(0,0,255),1)

    cv2.imshow("masked", black_image)
    cv2.waitKey(0)

    cv2.imshow("Result", result_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()