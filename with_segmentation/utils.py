import cv2
import numpy as np
from scipy.interpolate import splprep, splev 


def remove_hole(mask):
    reversed_mask = 255 - mask
    contours, _ = cv2.findContours(reversed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (h, w) = mask.shape
    black_image = np.zeros(mask.shape, np.uint8)
    
    n_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / (h * w) > 0.1: continue
        # contour_length = cv2.arcLength(contour, closed=True)
        # rate_length = min(contour_length / w, contour_length)
        # print(rate_area, rate_length)
        # if rate_area < 0.2 or rate_length > 6: continue
        n_contours.append(contour)

    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)

    mask = cv2.bitwise_or(mask, black_image)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel=kernel)

    # cv2.imshow("remove_hole", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)

    return mask


def make_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_image = np.zeros(mask.shape, np.uint8)

    n_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
      
        # contour_length = cv2.arcLength(contour, closed=True)
        # rate_length = min(contour_length / w, contour_length)
        # print(rate_area, rate_length)
        # if rate_area < 0.2 or rate_length > 6: continue

        x,y = contour.T
        x = x.tolist()[0]
        y = y.tolist()[0]
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 50)
        x_new, y_new = splev(u_new, tck, der=0)
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        n_contours.append(np.asarray(res_array, dtype=np.int32))

    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)
    
    # cv2.imshow("make_polygon", black_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)

    return black_image

def dilate_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel=kernel)

    # cv2.imshow("dilate_mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)

    return mask

def remove_others(previous_mask):
    kernel = np.ones((7, 7), np.uint8)
    temp_mask = cv2.dilate(previous_mask, kernel=kernel)
    (h, w) = previous_mask.shape
    black_image = np.zeros(previous_mask.shape, np.uint8)
    
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = []
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area / (h * w ) < 0.05:
            continue
        n_contours.append(contour)

    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)

    result = cv2.bitwise_and(previous_mask, black_image)
    # cv2.imshow("temp", result)
    # cv2.imshow("remove others", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)
    return result

def detect_straight(image):
    (h, w) = image.shape

    black_image = np.zeros(image.shape, np.uint8)
    contours, _ = cv2.findContours(image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    n_contours = []
    for cnt in contours:
        temp = np.zeros(image.shape, np.uint8)
        cv2.drawContours(temp, [cnt], 0, 255, -1)
        
        laplacian = cv2.Laplacian(temp,cv2.CV_8UC1)
        cst = cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)
        
        minLineLength = w // 8
        maxLineGap = 10

        lines = cv2.HoughLinesP(laplacian, 0.1, np.pi / 180, 1, None, minLineLength, maxLineGap)
        laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        if str(type(lines)) == "<class 'NoneType'>": continue
        contour = cv2.convexHull(cnt)
        n_contours.append(contour)
        # for line in lines:    
        #     for [x1,y1,x2,y2] in line:
        #         cv2.line(laplacian,(x1,y1),(x2,y2),(0,255,0),2)
             
        # cv2.imshow("1", laplacian)
        # cv2.waitKey(0)
    # temp_mask = cv2.bitwise_or(image, 255 - temp_mask)
    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)
    return black_image


def remove_small_contour(tmp):
    contours, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_image = np.zeros(tmp.shape, np.uint8)

    n_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000 :
            n_contours.append(contour)

    cv2.drawContours(black_image, tuple(n_contours), -1, (255,255,255), cv2.FILLED)
    # cv2.imshow("remove_small_contour", black_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)

    return black_image


def get_premask(temp):
    result = []
    result_image = np.zeros(temp[0].shape, np.uint8)
    black_image = np.zeros(temp[0].shape, np.uint8)
    
    i = 0
    for tmp in temp:
        i += 1
        black_image = remove_small_contour(tmp)
        # black_image = detect_straight(black_image)

        # contours,_ = cv2.findContours(black_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        # black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel=kernel)

        # for cnt in contours:
        #     cv2.drawContours(black_image,[cnt],0,255,-1)
        # temp_mask = cv2.bitwise_or(black_image, 255 - temp_mask)

        result_image = cv2.bitwise_or(result_image, black_image)
        # result.append(black_image)
        # if np.count_nonzero(black_image):
        #     cv2.imshow("mask" + str(i) + str(i), black_image)

    # kernel = np.ones((3, 3), np.uint8)
    # result_image = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, kernel=kernel)
    
    # cv2.waitKey(0)
    
    # max = 0
    # for tmp in result:
    #     if max < np.count_nonzero(tmp):
    #         black_image = tmp
    #         max = np.count_nonzero(tmp)
    # kernel = np.ones((5, 5), np.uint8)
    # black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel=kernel)
    # cv2.imshow("Premask", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(0)
    return result_image

def split_color(image):
    b, g, r =cv2.split(image)

    # cv2.imshow("b", b)
    # cv2.imshow("g", g)
    # cv2.imshow("r", r)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    b = cv2.bilateralFilter(b, 5, 75, 75)
    g = cv2.bilateralFilter(g, 5, 75, 75)
    r = cv2.bilateralFilter(r, 5, 75, 75)
    
    b = cv2.blur(b, (19, 19))
    g = cv2.blur(g, (19, 19))
    r = cv2.blur(r, (19, 19))

    

    g = g.astype(np.int16)
    b = b.astype(np.int16)
    r = r.astype(np.int16)

    return b, g, r

def get_colormask(b, g, r, mask_info):
    mask1, mask2 = g - r, b - g
    t1_mask1 = mask1 > mask_info[0][0] - 3
    t2_mask1 = mask1 < mask_info[0][1] + 3
    mask1 = np.bitwise_and(np.array(t1_mask1), np.array(t2_mask1)).astype(np.uint8) * 255

    t1_mask2 = mask2 > mask_info[1][0] - 3
    t2_mask2 = mask2 < mask_info[1][1] + 3
    mask2 = np.bitwise_and(np.array(t1_mask2), np.array(t2_mask2)).astype(np.uint8) * 255

    # if np.count_nonzero(mask1) != 0:
    #     cv2.imshow("mask1", mask1)
    #     cv2.imshow("mask2", mask2)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return mask1, mask2

def get_mask(image, mask_info, range_info):
    # preview = image.copy()
    # (h, w, _) = image.shape
    b, g, r = split_color(image)

    mask1, mask2 = get_colormask(b, g, r, mask_info=mask_info)
    range_mask = cv2.inRange(image, np.array(range_info[0]) - 10, np.array(range_info[1]) + 10)

    mask = cv2.bitwise_and(mask1, mask2) 
    mask = cv2.bitwise_and(mask, range_mask)

    # if np.count_nonzero(mask) != 0:
    #     cv2.imshow("range_mask", range_mask)
    #     cv2.imshow("mask", mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    return mask

def get_keypoints(masked_image, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    gray = cv2.blur(gray, ksize=(5, 5))
    adt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=7, C=1)
    adt_thresh = 255 - cv2.bitwise_or(adt_thresh, 255 - mask)

    contours, _ = cv2.findContours(adt_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    (h, w) = adt_thresh.shape
    black_image = np.zeros(adt_thresh.shape, np.uint8)
    rate_area = 0
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        x, y, ww, hh = cv2.boundingRect(contour)
        rate_area = area / (w * h)
        

        if rate_area < 0.001: 
            cv2.drawContours(black_image, [contour], 0, 255, -1)

    return black_image, 0

def temp_masks(image_path, mask_info_group, range_info_group):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1830, 918))
    (h, w, _) = image.shape
    
    image = cv2.resize(image, ( w // 3,h // 3))
    
    temp = []
    for i, mask_info in enumerate(mask_info_group):
        range_info = range_info_group[i]
        
        if range_info[0][0] < 60 and range_info[0][1] < 60 and range_info[0][2] < 60: continue
        mask = get_mask(image, mask_info, range_info)

        if np.count_nonzero(mask) / (h * w) < 0.0005: continue

        temp.append(mask)
    return image, temp

def get_keypoints(masked_image, mask):
    (h, w, _) = masked_image.shape
    gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
    # gray_hough = gray.copy()
    # corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    # corners = np.int0(corners)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(masked_image,(x,y),3,255,-1)

    laplacian = cv2.Canny(gray, 30, 50)
    # laplacian = cv2.Laplacian(gray,cv2.CV_8UC1)
    # cst = cv2.cvtColor(gray_hough,cv2.COLOR_GRAY2BGR)
    
    # minLineLength = w // 10
    # maxLineGap = 30

    # lines = cv2.HoughLinesP(gray, 0.1, np.pi / 180, 1, None, minLineLength, maxLineGap)
    # laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    # if str(type(lines)) == "<class 'NoneType'>": return masked_image

    # for line in lines:    
    #     for [x1,y1,x2,y2] in line:
    #         cv2.line(laplacian,(x1,y1),(x2,y2),(0,255,0),2)
            
    # cv2.imshow("1", laplacian)
    # cv2.waitKey(0)

    return laplacian