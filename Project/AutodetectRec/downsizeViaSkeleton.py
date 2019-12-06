import cv2
import numpy as np
import skeleton

def skeleton_downsize(img, factor=12):
    """Downsize while minizing information lost via skeleton downsize"""
    # perform skeleton
    h, w = img.shape
    skel = skeleton.skeleton(img)
    
    #downsize
    y_axis = ((np.where(skel==255))[0]/factor)
    x_axis = ((np.where(skel==255))[1]/factor)
    ret_h = h/factor
    ret_w = w/factor
    ret = np.zeros((int(ret_h), int(ret_w)))

    # upsample
    for i in range(len(y_axis)):
        ret[int(y_axis[i]), int(x_axis[i])] = 255
    return ret

if __name__ == "__main__": 
    pest = cv2.imread('pest_bl.png', 0)
    cv2.imwrite('skel_ds20.png', skeleton_downsize(pest, 20))

