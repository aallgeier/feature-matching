import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)

SOBEL_Y_KERNEL = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).astype(np.float32)

def compute_gauss_2d(ksize, sigma):
    """
    Returns a 2d discrete gaussian kernel.
    Args:
        ksize: kernel size
        sigma: standard deviation
    Returns:
        ksize x ksize discrete Gaussian kernel
    """
    center = int(ksize / 2)
    idxs = torch.arange(ksize).float()
    exponents = -((idxs - center) ** 2) / (2 * (sigma ** 2))
    gauss_1d = torch.exp(exponents)

    # Make 1d kernel entries sum to 1
    gauss_1d = gauss_1d.reshape(-1, 1) / gauss_1d.sum()
    gauss_2d = gauss_1d @ gauss_1d.T
    return gauss_2d.numpy()


def compute_image_gradients(img_gray):
    """Use convolution with Sobel filters to compute the first derivative of an image.

    Args:
        img_gray: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """
    # Paddings to keep image dimensions
    px = (len(SOBEL_X_KERNEL)-1) //2
    py = (len(SOBEL_Y_KERNEL)-1) //2

    # Manipulate input dimensions to meet specifications of nn.functional.conv2d().
    # kernel: (out_channels, in_channels, k_h, k_w), assuming groups is 1.
    kernel_X = torch.tensor(SOBEL_X_KERNEL).unsqueeze(0).unsqueeze(0)
    kernel_Y = torch.tensor(SOBEL_Y_KERNEL).unsqueeze(0).unsqueeze(0)
    # image: (batch, in_channels, H, W)
    img_gray = torch.tensor(img_gray).unsqueeze(0).unsqueeze(0)

    # Convolution. The convolution output has shape (minibatch, out_channels, h, w)
    # soremove the first two dimensions before returning.
    Ix = torch.squeeze(nn.functional.conv2d(img_gray, kernel_X, padding = px)).numpy()
    Iy = torch.squeeze(nn.functional.conv2d(img_gray, kernel_Y, padding = py)).numpy()

    return Ix, Iy

def compute_approx_auto_correlation(Ix, Iy, alpha, patch_size=7, sigma=5):
    """
    Compute Ix, Iy, IxIy in Szeliski (eq 7.8)
    and convolve with a window function weighting the pixel values
    within the window/patch where we are looking for corners.

    Args: 
        Ix, Iy: gradients of grayscale image
    """

    # Getting information of gradients at each pixel location
    # through element-wise multiplication.

    IxIx = Ix * Ix
    IyIy = Iy * Iy
    IxIy = Ix * Iy

    # Consider iamge patches centered at each pixel
    window_function = compute_gauss_2d(patch_size, sigma)

    # no padding since it may create corners
    R = np.zeros(Ix.shape)
    for x in range(patch_size//2, len(Ix[0]) - patch_size//2 - 1):
        for y in range(patch_size//2, len(Ix) - patch_size//2 - 1):
            
            # Smooth patch with gaussian
            patch_IxIx = window_function * IxIx[y - patch_size//2:y + patch_size//2+1, x - patch_size//2:x + patch_size//2+1]
            patch_IxIy = window_function * IxIy[y - patch_size//2:y + patch_size//2+1, x - patch_size//2:x + patch_size//2+1]
            patch_IyIy = window_function * IyIy[y - patch_size//2:y + patch_size//2+1, x - patch_size//2:x + patch_size//2+1]

            # Entries of autocorrelation matrix
            a = np.sum(patch_IxIx)
            b = np.sum(patch_IxIy)
            c = b
            d = np.sum(patch_IyIy)

            # Needed for computing score
            det = (a * d) - (b * c)
            trace = a + d

            # Harris score
            R[y, x] = det - alpha * (trace**2)

    return R

def maxpool(R, ksize):
    """
    For each pixel, assign the max value within its ksize x ksize window.
    """

    m, n = R.shape

    # taking max so padding with -inf  
    R_padded = np.pad(R, (((ksize - 1) // 2,), ((ksize - 1) // 2,)), 'constant', constant_values= -float("inf"))
    
    maxpooled_R = np.zeros(R.shape)
    for i in range(ksize//2, m - ksize//2 - 1):
        for j in range(ksize//2, n - ksize//2 - 1):
            maxpooled_R[i, j] = np.max(R_padded[i - ksize//2:i + ksize//2+1, j - ksize//2:j + ksize//2+1])
    
    return maxpooled_R


def top_k_interest_points(R, k, ksize):
    """
    Args:
        harris_score_map - (M, N) array with Harris corner score at each pixel
        k: num interest points sorted by confidence score
        ksize: kernel size for max-pooling operator
    
    """
    # Supress scores below median 
    # (this removes coreners in clouds for Notre Dame, for example)
    median = np.median(R)
    R[R < median] = 0

    # normalize image
    R = (R - np.min(R))/(np.max(R) - np.min(R))

    # assign each pixel the local maximum
    R_maxPooled = maxpool(R, ksize)

    # find pixels which are local maximum 
    scores = (R_maxPooled == R) * R

    scrores_sorted_ind = np.argsort(-scores.flatten())
    scores_sorted = -np.sort(-scores.flatten())

    # scores of the top k confidences
    c = scores_sorted[:k]

    # Get the x, y coordinates of the top k scores
    x, y = [], []
    for ind in scrores_sorted_ind[:k]:
        y.append(ind // scores.shape[1])
        x.append(ind %  scores.shape[1])
    
    x = np.array(x)
    y = np.array(y)

    return x, y, c

def remove_border_harris_points(img, x, y, c):
    """
    Remove interest points which are too close to the boarder.
    """

    h, w = img.shape
    x_valid = (8 <= x) & (x <= w - 8)
    y_valid = (8 <= y) & (y <= h - 8)

    valid_inds = x_valid * y_valid
    x, y, c = x[valid_inds], y[valid_inds], c[valid_inds]

    return x, y, c

def get_harris_corners(img_gray, k=2500):
    """
    Args:
        img_gray: black and white image
        k: number of interest points we will find at most 
    
    """

    ksize = 7
    alpha = 0.06

    Ix, Iy = compute_image_gradients(img_gray)
    R = compute_approx_auto_correlation(Ix, Iy, alpha)

    x, y, c = top_k_interest_points(R, k, ksize)
    x, y, c = remove_border_harris_points(img_gray, x, y, c)

    return x, y, c


    













