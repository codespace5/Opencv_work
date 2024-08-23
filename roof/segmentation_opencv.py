import cv2
import numpy as np

def _main():
    img = cv2.imread('roof.png')
    (h, w, _) = img.shape
    IMAGE_SHAPE = (h, w)
    lower = np.array([110, 110, 110])
    upper = np.array([180, 180, 180])
    red_mask = cv2.inRange(img, lower, upper)
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,50,150,apertureSize = 3)
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel=kernel)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=kernel)
    
    
    
    
    
    # kernel = np.ones((7, 7), np.uint8)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=kernel)
    
    # kernel = np.ones((17, 17), np.uint8)
    # final_opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel=kernel)
    
    contours, _  = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    temp_img = np.ones(IMAGE_SHAPE, dtype="uint8")
    img2 = img.copy()
    houses = []
    for cnt in contours:
        #convex_cnt = cv2.convexHull(cnt)
        area_cnt = np.count_nonzero(cnt)
        if area_cnt / h / w < 0.001:
            continue
        cv2.drawContours(temp_img, [cnt], 0, 255 ,-1)
        (x,y,i_w,i_h) = cv2.boundingRect(cnt)
        
        if x < 20 :
            d_x1 = 0
        else:
            d_x1 = x - 20
            
        if y < 20 :
            d_y1 = 0
        else:
            d_y1 = y - 20
            
        if x + i_w + 20 > w:
            d_x2 = w
        else:
            d_x2 = x + i_w + 20
            
        if y + i_h + 20 > h:
            d_y2 = h
        else:
            d_y2 = y + i_h + 20
        
        
        cv2.rectangle(img2, (d_x1,d_y1), (d_x2,d_y2), (255, 0, 0), 2)
        
        houses.append([img[d_y1: d_y2 , d_x1: d_x2], temp_img[d_y1: d_y2 , d_x1: d_x2], [d_x1, d_x2, d_y1, d_y2]])
        
    i = 5
    for tmp in houses:
        i += 1
        [house, mask, coord] = tmp
        # lap_mask = cv2.Laplacian(mask,cv2.CV_8UC1)
        # gray = cv2.cvtColor(house,cv2.COLOR_BGR2GRAY)
        # minLineLength = 100
        # maxLineGap = 10
        # lines = cv2.HoughLinesP(mask,1,np.pi/180,100,minLineLength,maxLineGap)
        # for line in lines:
        #     x1,y1,x2,y2 = line[0]
        #     cv2.line(house,(x1,y1),(x2,y2),(0,255,0),1)
        
        cv2.imshow(str(i), mask)
        i += 1
        
        # k = 2
        # data = np.float32(house).reshape((-1))
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        # ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # center = np.uint8(center)
        # result = center[label.flatten()]
        # center = sorted(center)
        # result[np.where(result == center[0][0])] = 255
        # result[np.where(result != 255)] = 0
        # result = result.reshape(house.shape)
        # cv2.imshow(str(i), result)\
    
    
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(temp_img, cv2.MORPH_CLOSE, kernel=kernel)
    
    cv2.imshow('3',img2)
    cv2.imshow('1',red_mask)
    cv2.imshow('2',closing)
    cv2.imwrite('detected.jpg',img2)
    cv2.imwrite("mask.jpg", red_mask)
    cv2.imwrite("closing.jpg", closing)
    cv2.waitKey(0)
    return

if __name__ == "__main__":
    _main()