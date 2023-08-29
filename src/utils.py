import cv2 
import numpy as np

def plot_interest_points(im_path, x, y):
    """
    Plot the interest points on an image

    Args:
        im_path: path to the rgb image
        x: x coordinates of interest points
        y: y coordinates of interest points
    
    """
    image = cv2.imread(im_path)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    for i in range(len(x)):
        color = np.random.rand(3) * 255
        image = cv2.circle(image, [x[i], y[i]], 5, color, -1)

    return image.astype(np.uint8)


def horizontal_stack_images(img1, img2):
    """
    horizontally stacking two images

    Args:
        img1, img2: rgb images
    """
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    h_stacked = np.zeros((max(h1, h2), w1+w2, 3))

    h_stacked[:h1, :w1] = img1
    h_stacked[:h2, w1:w1+w2] = img2

    return h_stacked

def plot_matches(im1, im2, x1, y1, x2, y2, matches):
    """
    x1, y1: interest points in image 1
    x2, y2: interest points in image 2

    matches: output from keypoint_match.match_features_ratio_test
    """

    hstacked_img = horizontal_stack_images(im1, im2)
    h1, w1, _ = im1.shape

    match_x1 = x1[matches[:, 0]]
    match_y1 = y1[matches[:, 0]]

    # needs to be shifted because the images are h-stacked
    match_x2 = x2[matches[:, 1]] + w1
    match_y2 = y2[matches[:, 1]]

    for i in range(len(matches)):
        start = (match_x1[i], match_y1[i])
        end = (match_x2[i], match_y2[i])

        color = np.random.rand(3) * 255

        # connect matching points with line
        hstacked_img = cv2.line(hstacked_img, start, end, color, 5)
        
        # plot matching circles
        hstacked_img = cv2.circle(hstacked_img, start, 10, color, -1)
        hstacked_img = cv2.circle(hstacked_img, end, 10, color, -1)

    
    return hstacked_img
  






