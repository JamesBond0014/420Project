import cv2
import numpy as np
from matplotlib import pyplot as plt

MONITORING = []
MAX_PER_FRAME_DIST = 100
STARTING_TEMPLATE = cv2.imread("template2.png")
# SCORE_CAP = 10.5 # neg inf to disable
# SCORE_CAP = -float("inf")
SCORE_CAP = 0.95

starting_frame = 18

vidcap = cv2.VideoCapture('test.mp4')

def reduce_locations(last_pos, max_loc, res, max_val, max_dist=MAX_PER_FRAME_DIST, verbose=False):
    # verbose = False
    # print(max_dist)
    previous_loc = (last_pos[1], last_pos[0])
    while abs((max_loc[0] - last_pos[1])) > max_dist or abs((max_loc[1] - last_pos[0])) > max_dist:
        res[max_loc[1], max_loc[0]] = 0
        if verbose:
            print("reprinting: " + str(max_loc) + "due to " + str(previous_loc))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if verbose:
        print("final: %s due to %s" % (str(max_loc), str(previous_loc)))
    return max_loc, max_val

def euclidean_distance(t1, t2):
    return ((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)**0.5

def template_matching(img, template, step=None, last_pos=None):

    img2 = img.copy()
    w, h, d = template.shape

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    methods_choice = [methods[3]] # Use method[3], this is the best

    for meth in methods_choice:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,STARTING_TEMPLATE,method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if step >= 191:
            speed = 25
        else:
            speed = 100

        if last_pos is not None:
            max_loc, max_val = reduce_locations(last_pos, max_loc, res, max_val, speed ,verbose=False)

        if max_val < SCORE_CAP:
            print("backup is attempted: %f at %d" % (max_val, step))
            res2 = cv2.matchTemplate(img,template,method)
            min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
            if last_pos is not None:
                max_loc2, max_val2 = reduce_locations(last_pos, max_loc2, res2, max_val2, speed , verbose=False)
            # if max_val2 > max_val:
            #     max_val = max_val2 
            #     print("backup is used")
            # else:
            #     print("backup is omitted")
            max_val = max_val2 
            print("backup is used (force)")
            
        if step in range(190,197):
            print(res.shape)
            cv2.imwrite('debug/%d.png' % step, (res/(1))*255)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            logging = min_val
        else:
            top_left = max_loc
            logging = max_val
        bottom_right = (top_left[0] + h, top_left[1] + w)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        new_template_suggest = img2[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
        # print(new_template_suggest.shape)

        # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(121),plt.imshow(img,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(new_template_suggest,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)

        # plt.show()

        # print(template.shape, new_template_suggest.shape)

        if step is not None:
            MONITORING.append((step, logging))

        return new_template_suggest, img, (top_left[1], top_left[0])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (852,480))
success,image = vidcap.read()

count = starting_frame + 1

curr_template = STARTING_TEMPLATE.copy()
new_loc = None

print(success)

# skip 17 frames
for i in range(starting_frame+1):
    vidcap.read()
success,image = vidcap.read()

while success:
    # cv2.imwrite("Bframe%d.jpg" % count, image)     # save frame as JPEG file      
    # print('Read a new frame: ', success)
    # cv2.imwrite("ConvoResults/Bframe%d_1src.jpg" % count, image)
    # cv2.imwrite("ConvoResults/Bframe%d_3previous_template.jpg" % count, curr_template)
    curr_template, result, new_loc = template_matching(image, curr_template, step=count, last_pos=new_loc)
    # if count in [17,201,182]:
    #     cv2.imshow('window', result)
    cv2.imwrite("ConvoResults/Bframe%d_2res.jpg" % count, result)
    cv2.imwrite("ConvoResults/Bframe%d_4updated_template.jpg" % count, curr_template)
    out.write(result)
    print(count)
    count += 1
    
    success,image = vidcap.read()

out.release()

for data in MONITORING:
    print(data[0], data[1])