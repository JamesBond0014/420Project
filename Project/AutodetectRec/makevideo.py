import cv2
import os
import numpy as np

import lineSlider
import CustomThreshold
import downsizeViaSkeleton
import perspective
import lineFromTwoPoints

MAX_PER_FRAME_DIST = 100
STARTING_TEMPLATE = cv2.imread("template2.png")
# SCORE_CAP = 10.5 # neg inf to disable
# SCORE_CAP = -float("inf")
SCORE_CAP = 0.95
vidcap = cv2.VideoCapture('test.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demonstration.avi',fourcc, 20.0, (360,880))

files = os.listdir('.')

for i in range(77):
    this_frame = cv2.imread("frame%d.jpg" % i)
    resized = cv2.resize(this_frame, ((360,880)))
    out.write(resized)

out.release()