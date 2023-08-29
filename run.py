import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image

import src.keypoint_descriptor as keypoint_descriptor 
import src.keypoint_match as keypoint_match
import src.utils as utils
import src.keypoint_detect as keypoint_detect

im_list = ["data/notredame1.jpg", "data/notredame2.jpg"]
features_list = []
x_list = []
y_list = []

for i, im_p in enumerate(im_list):
    im = np.array(Image.open(im_p), dtype=np.float32)
    im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x, y, c = keypoint_detect.get_harris_corners(im_gray)

    x_list.append(x)
    y_list.append(y)

    print("Num interest points:", im_p, len(x))
    
    descriptor = keypoint_descriptor.get_SIFT_descriptors(im_gray, x, y)
    features_list.append(descriptor)

matches, confidences = keypoint_match.match_features_ratio_test(features_list[0], features_list[1])

print("Num matches", len(matches))

im1 = np.array(Image.open(im_list[0]), dtype=np.float32)
im1 = cv2.resize(im1, (im1.shape[1]//2, im1.shape[0]//2))
im2 = np.array(Image.open(im_list[1]), dtype=np.float32)
im2 = cv2.resize(im2, (im2.shape[1]//2, im2.shape[0]//2))

plt.imshow(utils.plot_matches(im1, im2, x_list[0], y_list[0], x_list[1], y_list[1], matches).astype(np.uint8))
plt.show()



