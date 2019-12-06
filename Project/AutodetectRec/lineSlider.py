import cv2
import numpy as np
from lineFromTwoPoints import *

def hLineSlider(img):
    """ Atm returns the top matched line """
    h, w = img.shape
    data = {}
    prog = 0
    # Slide though some horizontal lines
    for i in range(h):
        p1 = (i, 0)
        for j in range(h):
            p2 = (j, w)
            line, length = line_from_two_points_with_len(p1, p2)
            score = calculate_convergance(img, line)/length
            data[score] = line
        prog += 1
        # print(prog, h)

    ranking = list(data)
    ranking.sort(reverse=True)
    top_score = ranking[0]
    top_line = data[top_score]
    
    # look for second line 
    
    i = 2
    second_line_tentative = data[ranking[1]]
    # Requirement: y_difference at least need to be 5 pixels apart
    y_diff = abs(top_line[0][0] - second_line_tentative[0][0])
    while y_diff < 5:
        second_line_tentative = data[ranking[i]]
        y_diff = abs(top_line[0][0] - second_line_tentative[0][0])
        i += 1
    
    return top_line, second_line_tentative

def line_explorer(img, interest_points):
    """ line_explorer, with bias at the bottom lower interest point"""

def v_line_finder(img, cz=100):
    # start by finding the patch of pixels that most likely have the line we want
    # use sum on axis 0

    # a = v_line_finder(thres_template)
    # b = array_chunk_sum(a)
    # c = max(b)
    # print(b.index(c))

    # step 1: find the v-line likeyhood by summing over axis 0
    likeyhood = img.sum(axis=0)
    # step 2: quantify data into chunksize of 100
    likey_range = array_chunk_sum(likeyhood, chunk_size=cz)
    range_pinned = likey_range.index(max(likey_range))
    # step 3: extend this range to size 200. Forward: 150, backward 50
    s = range_pinned-(cz//2)
    t = range_pinned+(cz//2) * 3

    return v_line_slider(img, s, t)


def array_chunk_sum(arr, chunk_size=100):
    """ helper for v_line_finder
        warning: make sure arr is bigger than chunk size
    """
    # initalization
    sum = 0
    results = []
    for i in range(chunk_size):
        sum += arr[i]
    results.append(sum)
    for i in range(len(arr)-chunk_size):
        sum -= arr[i]
        sum += arr[i+chunk_size]
        results.append(sum)
    return results

def v_line_slider(img, s, t):
    """ Another helper function for v_line_finder
        complexity: n^2
    """
    h, w = img.shape
    max_score = 0
    best_line = None

    # t is known bigger than s
    diff = t-s
    for i in range(diff):
        try_s = s+i
        pt_s = (0, try_s)
        for j in range(diff):
            try_t = s+j
            pt_t = (h, try_t)
            # print("forming line...")
            try_line, length = line_from_two_points_with_len(pt_s, pt_t)
            # print("formed line!")
            # print("calculating score...")
            score = calculate_convergance(img, try_line)/(length)
            if score > max_score:
                best_line = try_line
                max_score = score
            # print(i, j)
    return best_line

def line_from_two_points_with_len(pt1, pt2, hyperextend=False):
    """Ported as another helper for v_line_slider"""
    # Slope = rise over run 
    rise = pt1[0] - pt2[0]
    run = pt1[1] - pt2[1]
    if run != 0:
        slope = rise/run
    else: 
        slope = rise * float("inf")
    # guide from one line to the second line
    all_points = list()

    # if abs(slope) is greater than 1: iterate though y. Otherwise, x 
    if abs(slope) > 1: 
        # Because the axis in iteration can only go upwards, mark the smaller
        # of the two in the axis as reference point. 
        length = abs(rise)
        if rise > 0:
            ref_pt = pt2
        else:
            ref_pt = pt1
        for i in range(abs(rise)):
            all_points.append((i + ref_pt[0], int(np.rint(i/slope) + ref_pt[1])))
    else: 
        # Simalair arguement as rise case
        length = abs(run)
        if run > 0:
            ref_pt = pt2
        else:
            ref_pt = pt1
            
        for i in range(abs(run)):
            all_points.append((int(np.rint(slope*i)) + ref_pt[0], i + ref_pt[1]))

    return all_points, length

def calculate_convergance(img, line):
    """Helper function for line slider"""
    h, w = img.shape
    score = 0
    for pixel_loc in line:
        # prevent out of range
        if pixel_loc[0] < h and pixel_loc[1] < w:
            if img[pixel_loc[0], pixel_loc[1]] == 255:
                # Score is awarded
                score += 1
        else:
            # Give out of range warning
            # print("Warning: calculate_convergance determined", 
            #     str(pixel_loc), 
            #     "is out of range!")
            pass

    return score

def track_all_three(img):
    """Warning: this function only works (well) on downsized skeleton
    imgs"""

    # first go though the left edges of the image; then 
    # the bottom edge. Will attempt to find a piviot point 
    # via tracking lines. 

    line1, line2 = hLineSlider(img)

    if line1[0][0] < line2[0][0]:
        top_line = line1
        bottom_line = line2
    else:
        top_line = line2
        bottom_line = line1
    
    # pixel = top_line[0]
    # i = 1
    # all_protentals = []
    # while img[pixel[0] + 1, pixel[1]] == 255:
    #     all_protentals.append(pixel)
    #     pixel = top_line[i]
    #     i += 1

    # if top_line[0][1] < top_line[-1][1]:
    #     left_end_pixel = top_line[0]
    # else: 
    #     left_end_pixel = top_line[-1]

    progress = 0
    best_score = 0
    best_line = None

    scannable_up, scannable_down = generate_left_wall_and_floor_points(img.shape)
    total_su = len(scannable_up)
    for pixel_i in scannable_up:
        for pixel_j in scannable_down:
            try_line, length = line_from_two_points_with_len(pixel_i, pixel_j)
            score = calculate_convergance(img, try_line)/(length)
            if score > best_score:
                best_line = try_line
                best_score = score
        progress += 1
        # print(progress, total_su)
    
    return top_line, bottom_line, best_line

def generate_left_wall_and_floor_points(img_shape):
    h, w = img_shape
    points_up = []
    points_down = []
    for i in range(w//2):
        points_up.append((0, i))
        points_down.append((h-1, i))
    for i in range(h//2):
        points_down.append((i + h//2 -1, 0))
    return points_up, points_down

def hyperextend_line(line):
    """ hyperextension experiment, not 100% accurate but gets the job done.
    To be improved. """
    point_a = line[0]
    point_b = line[-1]
    delta_y = point_a[0] - point_b[0]
    delta_x = point_a[1] - point_b[1]

    extend_neg = (point_a[0] + delta_y, point_a[1] + delta_x)
    extend_pos = (point_b[0] - delta_y, point_b[1] - delta_x)

    # print(extend_neg)
    # print(extend_pos)

    far_left = line_from_two_points(extend_neg, point_a)
    far_right = line_from_two_points(point_b, extend_pos)
    ret = []
    ret.extend(far_left)
    ret.extend(line)
    ret.extend(far_right)
    return ret


if __name__ == "__main__":

    # Test hyperextension
    test = cv2.imread("hyperextensiontestbase.png")
    points = take_point_from_img(test)
    line = line_from_two_points(points[0], points[1])
    cv2.imwrite("normal_short_line.png", plot_points_to_img(test, line))

    k = hyperextend_line(line)

    print(line)
    print("")
    print(k)

    cv2.imwrite("hyperextended.png", plot_points_to_img(test, k))

    # USE_DOWNSIZE12 = True

    # org_img = cv2.imread("test_blacklist.jpg")

    # # chunk size vs factor:     100 for x1
    # #                           10 for x12
    # #                           5  for x20

    # # if USE_DOWNSIZE12:
    # #     thres_template = cv2.imread("skel_ds.png", 0)
    # #     print(v_line_finder(thres_template, 10))
    # # else: 
    # #     thres_template = cv2.imread("skel_ds20.png", 0)
    # #     print(v_line_finder(thres_template, 5))

    # h_slide_tester = cv2.imread("frame0.jpg", 0)
    # # print(hLineSlider(h_slide_tester)[1])
    # print(track_all_three(h_slide_tester)[2])