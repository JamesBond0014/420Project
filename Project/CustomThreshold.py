import cv2
import numpy as np
import os
# from matplotlib import pyplot as plt

# Threshold design tuned custom to the green grass of 
# soccer games verses the (almost) pure white lines. 

# looking at the grass fields: 
# Darkmost Gray ->      112 129 126
# Lightmost Green ->    94  121 54
#                        B   G  R

# Want some threshold that is defintely better than the lightmost green.
# Since green is hard to cut cleanly, compare threshold using B & R
def custom_threshold(img, b_thres=100, r_thres=120, 
                        use_pessimistic=True, absolute=True):
    """threshold the image with only blue and red. Green omitted
    img should therefore be three channeled. 
     """

    h, w, d = img.shape

    b_passes = img[:,:,0] > b_thres
    r_passes = img[:,:,2] > r_thres

    if use_pessimistic: final = b_passes & r_passes
    else: final = b_passes | r_passes

    if absolute: return final
    else: return (final * 255)

if __name__ == "__main__":
    tester_img = cv2.imread("test_blacklist.jpg")
    # print(tester_img)
    opts = custom_threshold(tester_img, use_pessimistic=False, absolute=False)
    pest = custom_threshold(tester_img, use_pessimistic=True, absolute=False)
    cv2.imwrite("pest_bl.png", pest)
    cv2.imwrite("opts_bl.png", opts)
