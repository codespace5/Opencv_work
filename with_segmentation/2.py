import cv2, os
import numpy as np   
from load_data import read_range_var
from utils import *

def _main():
    root_dir = 'test'
    
    image_names = os.listdir(root_dir)
    mask_info_group, range_info_group = read_range_var("Book1.xlsx")


    for image_name in image_names:

        image_path = os.path.join(root_dir, image_name)
        image, temp = temp_masks(image_path, mask_info_group, range_info_group)
        # def temp_masks(image_path, mask_info_group, range_info_group):
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (1830, 918))
        # (h, w, _) = image.shape
        
        # image = cv2.resize(image, ( w // 3,h // 3))
        
        # temp = []
        # for i, mask_info in enumerate(mask_info_group):
        #     range_info = range_info_group[i]
            
        #     if range_info[0][0] < 60 and range_info[0][1] < 60 and range_info[0][2] < 60: continue
        #     mask = get_mask(image, mask_info, range_info)


                    # def get_mask(image, mask_info, range_info):
                    #     # preview = image.copy()
                    #     # (h, w, _) = image.shape
                    #     b, g, r = split_color(image)

                    #     mask1, mask2 = get_colormask(b, g, r, mask_info=mask_info)
                    #     range_mask = cv2.inRange(image, np.array(range_info[0]) - 10, np.array(range_info[1]) + 10)

                    #     mask = cv2.bitwise_and(mask1, mask2) 
                    #     mask = cv2.bitwise_and(mask, range_mask)

                    #     # if np.count_nonzero(mask) != 0:
                    #     #     cv2.imshow("range_mask", range_mask)
                    #     #     cv2.imshow("mask", mask)
                    #     #     cv2.waitKey(0)
                    #     #     cv2.destroyAllWindows()
                    #     return mask

        #     if np.count_nonzero(mask) / (h * w) < 0.0005: continue

        #     temp.append(mask)
        # return image, temp

        if temp == []: continue
        mask = get_premask(temp)
            # def get_premask(temp):
            #     result = []
            #     result_image = np.zeros(temp[0].shape, np.uint8)
            #     black_image = np.zeros(temp[0].shape, np.uint8)
                
            #     i = 0
            #     for tmp in temp:
            #         i += 1
            #         black_image = remove_small_contour(tmp)
            #         # black_image = detect_straight(black_image)

            #         # contours,_ = cv2.findContours(black_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            #         # black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel=kernel)

            #         # for cnt in contours:
            #         #     cv2.drawContours(black_image,[cnt],0,255,-1)
            #         # temp_mask = cv2.bitwise_or(black_image, 255 - temp_mask)

            #         result_image = cv2.bitwise_or(result_image, black_image)
            #         # result.append(black_image)
            #         # if np.count_nonzero(black_image):
            #         #     cv2.imshow("mask" + str(i) + str(i), black_image)

            #     # kernel = np.ones((3, 3), np.uint8)
            #     # result_image = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, kernel=kernel)
                
            #     # cv2.waitKey(0)
                
            #     # max = 0
            #     # for tmp in result:
            #     #     if max < np.count_nonzero(tmp):
            #     #         black_image = tmp
            #     #         max = np.count_nonzero(tmp)
            #     # kernel = np.ones((5, 5), np.uint8)
            #     # black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel=kernel)
            #     # cv2.imshow("Premask", result_image)
            #     # cv2.waitKey(0)
            #     # cv2.destroyAllWindows(0)
            #     return result_image




        mask = remove_others(mask)
        mask = dilate_mask(mask, 7)
        mask = make_polygon(mask)
        mask = remove_hole(mask)
        mask = dilate_mask(mask, 10)


        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_image = cv2.bitwise_and(image, mask)
        
        keypoint_image = get_keypoints(masked_image, mask)
        # cv2.imshow("result", mask)

        

        cv2.imshow(image_name, image)
        cv2.imshow("mask", mask)
        cv2.imshow("masked_image", masked_image)
        cv2.imshow("keypoint_image", keypoint_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _main()
