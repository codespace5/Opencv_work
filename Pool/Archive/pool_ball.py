import cv2, os
from tkinter.filedialog import askopenfilenames, askdirectory
import numpy as np
import json

[133, 133, 109]

p1 = [274, 219]
p2 = [933, 239]
p3 = [927, 564]
p4 = [263, 548]

def detect_ball(image_path):
    image  = cv2.imread(image_path)
    lower = np.array([80, 100, 100])       #color mask
    upper = np.array([130, 160, 160])
    green_mask = cv2.inRange(image, lower, upper)
    green_mask = 255 - green_mask

    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_ERODE, kernel=kernel)

    contours,hierarchy = cv2.findContours(green_mask, 1, 1)
    black_image = np.zeros(image.shape, np.uint8)
    n_contours = []
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area < 50: continue

        if w > 30 or h > 30 : continue
        if w < 5 or h < 5 : continue

        if x < 263  :continue
        if x > 933  : continue
        if y < 219  : continue
        if y > 550  : continue

        centers.append([x + w // 2, y + h // 2])
            
        tmp = cv2.convexHull(contour)
        n_contours.append(tmp)

    pools = {}
    pools["balls_coordinates"] = centers
    return pools
    # cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)
    # black_image = cv2.dilate(black_image, kernel=kernel)



    # cv2.imshow("1", green_mask)
    # cv2.waitKey(0)
    # image = image - green_mask

    # balls = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,1,1,param1=30,param2=30, minRadius=3, maxRadius=15)
    # # if balls == []:
    # #     return
    # print(balls)
    # if balls is None:
    #     return
    # balls = np.uint16(np.around(balls))
    
    # for i in balls[0,:]:
    #     cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)

    cv2.imshow("1", black_image)
    cv2.waitKey(0)
    # print(balls)

def detect_white_pool(image_path):
    image  = cv2.imread(image_path)
    lower = np.array([190, 190, 190])       #color mask
    upper = np.array([255, 255, 255])
    green_mask = cv2.inRange(image, lower, upper)
    # green_mask = 255 - green_mask

    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel=kernel)

    contours,hierarchy = cv2.findContours(green_mask, 1, 1)
    
    black_image = np.zeros(image.shape, np.uint8)
    n_contours = []
    white = []
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area < 90: continue
        if area > 200: continue
        if max_area < area:
            white = [x + w // 2, y + h // 2]
            
        # tmp = cv2.convexHull(contour)
        n_contours.append(contour)

    # pools = {}
    # pools["balls_coordinates"] = centers
    # return pools
    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)
    black_image = cv2.dilate(black_image, kernel=kernel)

    cv2.imshow("1", green_mask)
    cv2.waitKey(0)

    return white

def _main():
    # files = askopenfilenames()
    root_dir = askdirectory()
    files = os.listdir(root_dir)

    for image_name in files:
        if "png" in image_name:
            image_path = os.path.join(root_dir, image_name)

            pools = detect_ball(image_path)
            white_pool = detect_white_pool(image_path)
            
            [pool_coords] = list(pools.values())
            if pool_coords == []:
                pool_coords = [white_pool]

            tmp = []
            for coords in pool_coords:
                if white_pool == [] : 
                    tmp.append(coords)
                    continue
                if abs(white_pool[0] - coords[0]) < 5 and abs(white_pool[1] - coords[1]):
                    tmp.append(white_pool)
                    continue
                tmp.append(coords)
            pool_coords = tmp

            pools = {}
            pools["balls_coordinates"] = pool_coords    
            if white_pool == [] :
                pools["cue_ball_index"] = "I"
            else:
                pools["cue_ball_index"] = white_pool

            with open(image_name.split(".")[0] + '.json', 'w') as fp:
                json.dump(pools, fp)

if __name__ == "__main__":
    _main()