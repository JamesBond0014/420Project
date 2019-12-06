import cv2
import numpy as np

proper_tester = cv2.imread("test_blacklist.jpg")
proper_tester = proper_tester[:,:,2] # I only want the red channel for my BW img
# print(proper_tester)
skel = cv2.imread("skel.png", 0)
pest = cv2.imread("pest.png", 0)

ret = cv2.cornerHarris(pest, 15, 3, 0.02)
ret -= ret.min()
# print(ret)
ret = ret / ret.max()
ret = ret * 255
# cv2.imshow("ret", ret)
print(ret.min(), ret.max())
# print(ret)
# cv2.waitKey()
cv2.imwrite('corner_results.png', ret)

# perform threshold on corner_results
# cv.THRESH_BINARY

r, ret = cv2.threshold(ret, 230, 255, cv2.THRESH_BINARY)
cv2.imwrite('thres_corner_results.png', ret)
# print(ret.sum())