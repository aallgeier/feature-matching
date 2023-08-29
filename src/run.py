import keypoint_detect as keypoint_detect
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import keypoint_descriptor
from utils import plot_interest_points

im_list = ["../data/notredame1.jpg", "../data/notredame2.jpg"]


cv2.startWindowThread()
for i, im_p in enumerate(im_list):
    im = np.array(Image.open(im_p), dtype=np.float32)
    im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x, y, c = keypoint_detect.get_harris_corners(im_gray)

    print("num interest points:", im_p, len(x))
    
    cv2.imshow(im_p, plot_interest_points(im_p, x, y))

    descriptor = keypoint_descriptor.get_SIFT_descriptors(im_gray, x[:100], y[:100])
    cv2.imshow("descriptor" + str(i), descriptor)

cv2.waitKey(0)
cv2.destroyAllWindows()



