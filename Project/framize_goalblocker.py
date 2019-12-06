import numpy as np
import os
import cv2

TARGET_P1 = (0, 187)
TARGET_P2 = (675, 495)

def blacklist(frame):
    cv2.rectangle(frame, TARGET_P1, TARGET_P2, (0, 0, 0), -1)

vid_name = 'pen6slow.mp4'
vidcap = cv2.VideoCapture(vid_name)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
success,image = vidcap.read()
frame_count = 0

folder_name = vid_name + str("_fbfb")
if folder_name not in os.listdir():
    os.mkdir(folder_name)

while success:
    blacklist(image)
    cv2.imwrite("%s/frame%d.jpg" % (folder_name, frame_count), image)
    print(frame_count)

    success,image = vidcap.read()
    frame_count += 1
