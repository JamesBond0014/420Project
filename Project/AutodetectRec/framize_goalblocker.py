import numpy as np
import os
import cv2

def goal_blocker(vid_name):
    # This is one rectangle 
    TARGET_P1 = (0, 187)
    TARGET_P2 = (675, 495)

    # second, remove ads
    TARGET_P3 = (0,0)
    TARGET_P4 = (1919, 70)

    def blacklist(frame):
        cv2.rectangle(frame, TARGET_P1, TARGET_P2, (0, 0, 0), -1)
        cv2.rectangle(frame, TARGET_P3, TARGET_P4, (0, 0, 0), -1)

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

if __name__ == "__main__":
    vid_name = 'pen6slow.mp4'
    goal_blocker(vid_name)