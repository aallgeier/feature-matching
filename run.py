import harris_corner
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

from utils import plot_interest_points


im = np.array(Image.open("notredame.jpg"), dtype=np.float32)
im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
x, y, c = harris_corner.get_harris_corners(im_gray)

plot_interest_points("notredame.jpg", x, y)

