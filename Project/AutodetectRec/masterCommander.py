import cv2
import os
import numpy as np

import lineSlider
import CustomThreshold
import downsizeViaSkeleton
import perspective
import lineFromTwoPoints

# Open gloabl file
ball_location = open("ball_location.txt", 'r')
ALL_BALL_LOCATIONS = ball_location.readlines()
for i in range(len(ALL_BALL_LOCATIONS)):
    ALL_BALL_LOCATIONS[i] = ALL_BALL_LOCATIONS[i].strip()
    ALL_BALL_LOCATIONS[i] = ALL_BALL_LOCATIONS[i].split(' ')
    ALL_BALL_LOCATIONS[i][0] = int(float(ALL_BALL_LOCATIONS[i][0]))
    ALL_BALL_LOCATIONS[i][1] = int(float(ALL_BALL_LOCATIONS[i][1]))

print(ALL_BALL_LOCATIONS)
LOCATIONS_PREVIOUS = []
LOCATION_DIFFERNECE = []
ball_location.close()

def per_frame_op(image, scale_factor=12, illustration_img=None, frame_number=0):
    if illustration_img is None:
        illustration_img = image.copy()
    h, w, d = image.shape
    # use pessmestic, not absolute
    cust_thres = CustomThreshold.custom_threshold(image, absolute=False)
    # downsize reasonabliy 
    cv2.imwrite("temp.png", cust_thres)
    a = cv2.imread("temp.png", 0)
    downsized = downsizeViaSkeleton.skeleton_downsize(a, factor=scale_factor)
    dh, dw = downsized.shape

    v1_line = lineSlider.v_line_finder(downsized,10)
    other_lines = lineSlider.track_all_three(downsized)

    a3 = np.ndarray((dh,dw,d))
    a3[:,:,0] = downsized
    a3[:,:,1] = downsized
    a3[:,:,2] = downsized

    # ret = lineFromTwoPoints.plot_points_to_img(a3, v1_line, (255,0,255))
    # ret = lineFromTwoPoints.plot_points_to_img(ret,other_lines[0], (255,255,0))
    # ret = lineFromTwoPoints.plot_points_to_img(ret,other_lines[1], (0,255,255))
    # ret = lineFromTwoPoints.plot_points_to_img(ret,other_lines[2], (0,255,0))

    # For now, we will check the intersection of two lines
    # indices: top_line, bottom_line, best_line

    # v1_line and bottom line:
    bottom_right = None
    for pixel in v1_line:
        if pixel in other_lines[1]:
            bottom_right = pixel
    if bottom_right is None:
        bottom_right = desperate_measures(v1_line, other_lines[1])
    # print(pixel)

    top_right = None
    for pixel in v1_line:
        if pixel in other_lines[0]:
            top_right = pixel
    if top_right is None:
        top_right = desperate_measures(v1_line, other_lines[0])

    top_left = None
    for pixel in other_lines[2]:
        if pixel in other_lines[0]:
            top_left = pixel
    if top_left is None:
        top_left = desperate_measures(other_lines[2], other_lines[0])

    # need hyperextension
    bottom_left = None
    hyper_bottom = lineSlider.hyperextend_line(other_lines[1])
    hyper_left = lineSlider.hyperextend_line(other_lines[2])
    for pixel in hyper_left:
        if pixel in hyper_bottom:
            bottom_left = pixel
    if bottom_left is None:
        bottom_left = desperate_measures(v1_line, other_lines[0])

    pts = [top_left, top_right, bottom_left, bottom_right]
    
    # need to restore the points into their factor sizes so perspective can be 
    # performed
    restored = []
    # scrtech up y direction a bit 
    for pt in pts:
        restored.append((pt[1]*scale_factor, pt[0]*scale_factor))
    coords = np.array(restored, dtype="float32")
    ih, iw, i_d = illustration_img.shape
    # distance_map = np.ndarray((ih * 2, iw, i_d))
    # distance_map[int(ALL_BALL_LOCATIONS[i][0]) * 2, int(ALL_BALL_LOCATIONS[i][1]), 0] = 255
    # distance_map[int(ALL_BALL_LOCATIONS[i][0]) * 2, int(ALL_BALL_LOCATIONS[i][1]), 1] = 255
    # distance_map[int(ALL_BALL_LOCATIONS[i][0]) * 2, int(ALL_BALL_LOCATIONS[i][1]), 2] = 255

    illustration_img_long = cv2.resize(illustration_img,(ih*2, iw))
    wraped = perspective.four_point_transform(illustration_img, coords)
    # distance_calc = perspective.four_point_transform(distance_map, coords)
    # point_location = lineFromTwoPoints.take_point_from_img(distance_calc)
    # if LOCATIONS_PREVIOUS != []:
    #     previous_location = LOCATIONS_PREVIOUS[-1]
    # else: 
    #     previous_location = point_location
    # LOCATIONS_PREVIOUS.append(point_location)
    # LOCATION_DIFFERNECE.append(abs(previous_location-point_location))
    # cv2.imshow('tester', wraped)
    # cv2.waitKey(
    return wraped


def desperate_measures(line1, line2):
    print("Warning: desperate measures")
    for pixel in line1:
        for alt_pixel in line2:
            if abs(pixel[0]-alt_pixel[0]) <= 1 and abs(pixel[1] - alt_pixel[1]) <= 1:
                return pixel
    else: raise Exception

def op_vid(vid_name):
    print("op_vid: starting.")
    # 1. read video, apply block

    # This is one rectangle 
    TARGET_P1 = (0, 187)
    TARGET_P2 = (675, 495)
    # second, remove ads
    TARGET_P3 = (0,0)
    TARGET_P4 = (1919, 70)
    vidcap = cv2.VideoCapture(vid_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    success,image = vidcap.read()
    image_org = image.copy()
    frame_count = 0

    folder_name = vid_name + str("_processed")
    if folder_name not in os.listdir():
        os.mkdir(folder_name)

    while success:
        image_org = image.copy()
        cv2.rectangle(image, TARGET_P1, TARGET_P2, (0, 0, 0), -1)
        cv2.rectangle(image, TARGET_P3, TARGET_P4, (0, 0, 0), -1)
        ret = per_frame_op(image, scale_factor=12, illustration_img=image_org)
        cv2.imwrite("%s/frame%d.jpg" % (folder_name, frame_count), ret)
        print(frame_count)
        success,image = vidcap.read()
        frame_count += 1


if __name__ == "__main__":
    vid_name = 'pen6slow.mp4'
    op_vid(vid_name)

    print("it appears that the ball has travelled a total of %s" % (sum(LOCATION_DIFFERNECE)))
    pass