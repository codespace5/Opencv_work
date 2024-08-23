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

        if temp == []: continue
        mask = get_premask(temp)
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
