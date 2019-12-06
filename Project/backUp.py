import cv2
import numpy as np
import itertools

def draw_line(img, pt1, pt2):
    pass

def plot_points_to_img(img, point_cloud, c3_color=(255,255,255)):
    """Plot some points to an image"""
    # Convert into 1 channel image if it's three channelled
    ret = img.copy()
    # if len(ret.shape) == 3:
    #     ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for pt in point_cloud:
        if len(ret.shape) == 2:
            ret[pt[0], pt[1]] = 255
        else: 
            ret[pt[0], pt[1]] = c3_color
    return ret

def plot_points_to_img_grad(img, point_cloud):
    """Plot some points to an image"""
    # Convert into 1 channel image if it's three channelled
    ret = img.copy()
    pixels = len(point_cloud)
    pt_count = 0
    grad_trend = 255/pixels
    if len(ret.shape) == 3:
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for pt in point_cloud:
        ret[pt[0], pt[1]] = 255 - pt_count * grad_trend
        pt_count += 1 
    return ret


def line_from_two_points(pt1, pt2, infinite=False):
    """Returns: a list of all points that constructs a line from pt1 to pt2"""
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
        if rise > 0:
            ref_pt = pt2
        else:
            ref_pt = pt1

        for i in range(abs(rise)):
            all_points.append((i + ref_pt[0], int(np.rint(i/slope) + ref_pt[1])))
    else: 
        # Simalair arguement as rise case
        if run > 0:
            ref_pt = pt2
        else:
            ref_pt = pt1
            
        for i in range(abs(run)):
            all_points.append((int(np.rint(slope*i)) + ref_pt[0], i + ref_pt[1]))

    return all_points

def take_point_from_img(img):
    # Convert into 1 channel image if it's three channelled
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = img.shape
    # points = set()
    # for i in range(h):
    #     for j in range(w):
    #         if img[i,j] == 255: points.add((i,j))

    result = np.where(img == 255)
    points = list(zip(result[0], result[1]))

    return points

if __name__ == "__main__":
    a = cv2.imread('vPoints.png')
    pt1, pt2 = (take_point_from_img(a))
    print(pt1, pt2)
    point_cloud = line_from_two_points(pt2,pt1)
    print(point_cloud)
    k = plot_points_to_img(a, point_cloud)
    cv2.imwrite('v.png', k)

    # a = cv2.imread('bigCliqueBase.png')
    # ret = a.copy()
    # pts = (take_point_from_img(a))
    # k = itertools.combinations(pts, 2)
    # up_to = 96580
    # count = 0
    # for comb in k:
    #     # point_cloud = line_from_two_points(comb[0],comb[1])
    #     point_cloud2 = line_from_two_points(comb[1],comb[0])
    #     # ret = plot_points_to_img_grad(ret, point_cloud)
    #     ret = plot_points_to_img_grad(ret, point_cloud2)
    #     # plot_points_to_img_grad(ret, point_cloud2)
    #     if not count%5000:
    #         print("%d/%d" %(count, up_to))

    #     count += 1
    # # ret[np.where(ret>255)] = 255
    # cv2.imwrite('bigCliqueBase.png', ret)

    # point_cloud_db = line_from_two_points((620, 873), (117, 835))
    # print(point_cloud_db)

