import cv2 
import numpy as np

def plot_interest_points(im_path, x, y):
    """
    Plot the interest points on an image

    Args:
        im_path - path to the rgb image
        x - x coordinates of interest points
        y - y coordinates of interest points
    
    """
    image = cv2.imread(im_path)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    for i in range(len(x)):
        color = np.random.rand(3) * 255
        image = cv2.circle(image, [x[i], y[i]], 5, color, -1)

    cv2.startWindowThread()
    cv2.imshow("image", (image).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
