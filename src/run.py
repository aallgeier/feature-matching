import src.keypoint_detect as keypoint_detect
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

from src.utils import plot_interest_points

im_list = ["notredame1.jpg", "notredame2.jpg"]

print(np.linspace(-np.pi, np.pi, 10))

# for im_p in im_list:
#     im = np.array(Image.open(im_p), dtype=np.float32)
#     im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))

#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     x, y, c = keypoint_detect.get_harris_corners(im_gray)

#     print("num interest points:", im_p, len(x))
#     cv2.startWindowThread()
#     cv2.imshow(im_p, plot_interest_points(im_p, x, y))

# cv2.waitKey(0)
# cv2.destroyAllWindows()



