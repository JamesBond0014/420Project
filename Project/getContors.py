import cv2
import numpy as np

im = cv2.imread('pest.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(type(contours))

illu = cv2.drawContours(im, contours, -1, (0,255,0), 3)
print(illu.shape)