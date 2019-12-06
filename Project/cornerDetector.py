import sys
import cv2
import numpy
import imutils
import matplotlib.pyplot as plt

def __normalize1ChannelImg(img_2d, tff=True):
    # Only normaizes if the image has 1 channel
    max_val = img_2d.max()
    min_val = img_2d.min()
    sum_val = max_val - min_val

    output = (img_2d-min_val)/sum_val
    if tff: 
        output = output*255
    return output

def normalize1ChannelImg(img_2d, tff=True):
    return __normalize1ChannelImg(img_2d, tff=True)


def __make3Channel(img_2d):
    height, width = img_2d.shape
    ret = numpy.ndarray((height, width, 3))
    for i in range(height):
        for j in range(width):
            ret[i,j,0] = img_2d[i,j]
            ret[i,j,1] = img_2d[i,j]
            ret[i,j,2] = img_2d[i,j]

    return ret


def harrisCorners(input_img, alpha=0.05, sigma=7, brown=False):
    # Harris: det(M) - aTrace(M)^2
    # 
    # M =   |Ix^2   IxIy|
    #       |IxIy   Iy^2|
    #
    
    #=============================== Harris R ================================
    # 1. Compute Ix Iy => Ix^2 Iy^2 IxIy
    ## 1.1 Grayscale image, blur
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), sigma)
    ## 1.2 Compute the values
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    IxIy = numpy.multiply(Ix, Iy)
    Ix2 = numpy.multiply(Ix, Ix)
    Iy2 = numpy.multiply(Iy, Iy)

    # 2. Blur the results
    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10)

    # 3. Use the computed value to calculate R.
    ## 3.1 det = Ix^2*Iy^2 - (IxIy)^2
    det = numpy.multiply(Ix2_blur, Iy2_blur) - numpy.multiply(IxIy_blur,IxIy_blur)
    ## 3.2 trace = Ix^2 + Iy^2
    trace = Ix2_blur + Iy2_blur
    ## R = det - alpha*trace^2

    if brown:
        # Sanitize zeros and change them to 0.001
        more_than = trace > -0.001
        less_than = trace < 0.001
        targets = more_than & less_than
        trace[targets] = 0.001
        # targets = numpy.isnan(trace)
        # trace[targets] = 0
        # targets = numpy.isnan(det)
        # det[targets] = 0
        ret = numpy.divide(det, trace)
        # ret = det/trace
        return ret
    
    ret = det - alpha * numpy.multiply(trace, trace)
    #================================= End ===================================

    return ret

def inverte(img):
    ret = (255-img)
    return ret 

def threshold_img(img, thres):
    # Apply a % threashold to the img, thres should be in between 0-100
    # Warning: mutates img

    min_val = img.min()
    threshold_actual = min_val + (img.max()-min_val)*thres

    super_threshold_indices = img < threshold_actual
    img[super_threshold_indices] = 0

def lower_threshold_img(img, thres):
    # Apply a % threashold to the img, thres should be in between 0-100
    # Warning: mutates img

    min_val = img.min()
    threshold_actual = min_val + (img.max()-min_val)*thres

    super_threshold_indices = img > threshold_actual
    img[super_threshold_indices] = 0


def __highlight_1channel_img(img_2d):
    # Takes a one channel image and highlight it

    height, width = img_2d.shape[0:2]
    highlighted_img = numpy.ndarray((height,width,3))
    for i in range(height):
        for j in range(width):
            highlighted_img[i,j,1] = img_2d[i,j]
            highlighted_img[i,j,0] = img_2d[i,j]
    return highlighted_img


def __zero_negatives(img):
    neg_indicies = img < 0
    img[neg_indicies] = 0


def __test_alpha_values(img):
    # algo that brute forces through 20 alpha values from 0.04-0.06 
    # And determines the best alpha value for the cause

    for i in range(40):
        this_alpha_int = 30 + i
        this_alpha = numpy.around(float(this_alpha_int) * 0.001, 3)
        corners = harrisCorners(img, this_alpha)
        # print(corners[128,128])
        __zero_negatives(corners) 
        corners_norm = __normalize1ChannelImg(corners)
        # threshold_img(corners_norm, 0.17)
        print(str(this_alpha)+": " + str(corners_norm.sum()))
        cv2.imwrite('alpha'+str(this_alpha)+'.png', __highlight_1channel_img(corners_norm))

# def __blob_fill(img_2d, i, j, visited=set()):
#     height, width = img_2d.shape
#     visited.add((i,j))
#     if i-1 >= 0 and (not (i-1, j) in visited) and img_2d[i-1, j] > 0:
#         __blob_fill(img_2d, i-1, j, visited)
#     if i+1 < height and (not (i+1, j) in visited ) and img_2d[i+1, j] > 0:
#         __blob_fill(img_2d, i+1, j, visited)
#     if j-1 >= 0 and (not (i, j-1) in visited) and img_2d[i, j-1] > 0:
#         __blob_fill(img_2d, i, j-1, visited)
#     if j+1 < width and (not (i, j+1) in visited) and img_2d[i, j+1] > 0:
#         __blob_fill(img_2d, i, j+1, visited)

def __blob_fill(img_2d, i, j):
    height, width = img_2d.shape
    visited = set()
    visited.add((i,j))
    OPEN = list()

    # Setup 
    if i-1 >= 0 and (not (i-1, j) in visited) and img_2d[i-1, j] > 0:
        OPEN.append((i-1, j))
    if i+1 < height and (not (i+1, j) in visited ) and img_2d[i+1, j] > 0:
        OPEN.append((i+1, j))
    if j-1 >= 0 and (not (i, j-1) in visited) and img_2d[i, j-1] > 0:
        OPEN.append((i, j-1))
    if j+1 < width and (not (i, j+1) in visited) and img_2d[i, j+1] > 0:
        OPEN.append((i, j+1))

    while OPEN:
        this_pixel = OPEN.pop()
        visited.add(this_pixel)
        i = this_pixel[0]
        j = this_pixel[1]
        if i-1 >= 0 and (not (i-1, j) in visited) and img_2d[i-1, j] > 0:
            OPEN.append((i-1, j))
        if i+1 < height and (not (i+1, j) in visited ) and img_2d[i+1, j] > 0:
            OPEN.append((i+1, j))
        if j-1 >= 0 and (not (i, j-1) in visited) and img_2d[i, j-1] > 0:
            OPEN.append((i, j-1))
        if j+1 < width and (not (i, j+1) in visited) and img_2d[i, j+1] > 0:
            OPEN.append((i, j+1))

    return visited

def NMS(img_2d, zeroed=True):
    __NMS(img_2d, zeroed)

def __NMS(img_2d, zeroed=True):
    # Note: Non Maxima Supression only works on 1 channel images
    if not zeroed:
        __zero_negatives(img_2d)
    height, width = img_2d.shape
    all_blobs = []
    all_visited_pixels = set()
    for i in range(height):
        for j in range(width):
            if img_2d[i,j] > 0:
                if not (i,j) in all_visited_pixels:
                    # type(this_blob) >>> set()
                    this_blob = list(__blob_fill(img_2d, i, j))
                    all_blobs.append(this_blob)
                    for pixel in this_blob:
                        all_visited_pixels.add(pixel)

    local_maxs = []
    for blob in all_blobs:
        this_max = 0
        max_pixel_location = (0,0)
        for pixel in blob:
            if this_max < img_2d[pixel]:
                this_max = img_2d[pixel]
                max_pixel_location = pixel
        local_maxs.append(max_pixel_location)

    supress_indicies = (img_2d == img_2d)
    for pixel_addr in local_maxs:
        supress_indicies[pixel_addr] = False
    
    img_2d[supress_indicies] = 0   

def __print_area(img_2d, targets):
    # Mark the area out from a 1 channel image 
    height, width = img_2d.shape
    ret = numpy.ndarray((height, width, 3))
    ret[:,:,2] = img_2d
    for target in targets:
        i, j = target
        ret[i,j,0] = 255
    return ret


def rotate_by_angle(img, angle):
    rotated = imutils.rotate_bound(img, angle)
    return rotated


if __name__ == "__main__":
    # colourTemplate = cv2.imread("colourTemplate.png")
    # gray = cv2.cvtColor(colourTemplate, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5), 7)
    # Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    # Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    # cv2.imwrite("blur.png", blur)

    # print(Ix[600,754]) #0
    # print(Ix[623,832]) #709.0
    # print(Ix[45,45]) #0
    # print(Ix[659,1227]) #-240.0
    # print(Ix.min(), Ix.max())
    # # print(Ix[600,754])
    # # print(Ix[600,754])
    # Ixn = __normalize1ChannelImg(Ix)
    # Ix3 = __make3Channel(Ixn)

    # cv2.imwrite("Ix.png", Ix)
    # cv2.imwrite("Ix3.png", Ix3)
    # cv2.imwrite("Iy.png", Iy)

    # print(Ix3.shape)
    # print(type(Ix))

    # plt.imshow(Ix,cmap = 'gray')

    building = cv2.imread("pest_bl.png")
    corners = harrisCorners(building, brown=False)
    cv2.imwrite('corners_raw.png', corners)
    print(corners.max()) 
    print(corners.min()) 
    __zero_negatives(corners)
    corners_norm = __normalize1ChannelImg(corners)
    # print(corners_norm[0,166])
    cv2.imwrite('rotated_harris.png', corners_norm)
    threshold_img(corners_norm, 0.05)
    cv2.imwrite('rotated_harris_thres.png', corners_norm)

    __NMS(corners_norm, zeroed=True)
    cv2.imwrite('rotated_harris_thres_NMS.png', corners_norm)
    count = corners_norm>0
    print(count.sum())

    # building_rotated = rotate_by_angle(building, -60)
    # cv2.imwrite('building_rotated.png', building_rotated)

    # # perform the same harris on building_rotated
    # corners_60 = harrisCorners(building_rotated)
    # __zero_negatives(corners_60)
    # corners_norm_60 = __normalize1ChannelImg(corners_60)
    # threshold_img(corners_norm_60, 0.05)
    # cv2.imwrite('corners_norm_thres_rot60.png', corners_norm_60)
    # __NMS(corners_norm_60, zeroed=True)
    # cv2.imwrite('corners_norm_thres_rot60_nms.png', corners_norm_60)
    # count = corners_norm_60>0
    # print(count.sum())
