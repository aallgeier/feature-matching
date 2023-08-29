import numpy as np
from src.keypoint_detect import compute_image_gradients

"""
The descriptor is based on the local image descriptor of SIFT
"""

def get_magnitudes_and_orientations(Ix, Iy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """
    magnitudes = []  # placeholder
    orientations = []  # placeholder

   # Use Pythagorean to get magnitudes. 
    magnitudes = np.sqrt(Ix**2 + Iy**2)

    # Get tangent values at each location and use arctan appropriately. 
    orientations = np.arctan2(Iy, Ix)

    return magnitudes, orientations

def get_histogram(
    window_magnitudes,
    window_orientations,
    patch_size: int = 16,
    size_grouped_cells: int = 4,
    num_bins: int = 8
    ):
    """ 

    Create a histogram of gradient orientations, weighted by the gradient 
    magnitudes.
    
    You might divide the patch into smaller sub-blocks (e.g., 4x4) and compute 
    histograms for each, concatenating them to form the final descriptor.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    weighted_histograms = np.zeros((size_grouped_cells**2, num_bins))

    l = patch_size//size_grouped_cells
    
    k = 0
    for i in range(size_grouped_cells):
        for j in range(size_grouped_cells):
            
            # Get magnitudes and orientations in 4x4 cell within the 16 x 16 window
            cell_magnitutdes = window_magnitudes[i*l:(i+1)*l, j*l:(j+1)*l].flatten()
            cell_orientations = window_orientations[i*l:(i+1)*l, j*l:(j+1)*l].flatten()

            # For this 4x4 window, create a histogram of orientations weighted by magnitudes.
            bins = np.linspace(-np.pi, np.pi, num_bins+2)
            histogram = np.histogram(np.around(cell_orientations, decimals=5), bins, weights= cell_magnitutdes)[0]
            
            weighted_histograms[k] = histogram
            k+=1

    return np.reshape(weighted_histograms, ((size_grouped_cells**2) * num_bins, 1))

def get_feat_vec(
    c: float,
    r: float,
    magnitudes,
    orientations,
    patch_size: int = 16,
    num_local_cells: int = 4,
    num_bins: int = 8
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.

    (1) Each feature is normalized to unit length.
    (2) Each feature is raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    Args:
        c: a float, the column (x-coordinate) of the interest point
        r: A float, the row (y-coordinate) of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: size of patch (i.e. 16)
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
    """

    # Variables to record the coordinates of the top-left corner of the window.
    w_top_left_c = c - (patch_size//2 - 1)
    w_top_left_r = r - (patch_size//2 - 1)

    # Get window magnitudes and orientations.
    window_magnitudes = magnitudes[w_top_left_r: w_top_left_r + patch_size, w_top_left_c:w_top_left_c + patch_size]
    window_orientations = orientations[w_top_left_r: w_top_left_r + patch_size, w_top_left_c:w_top_left_c + patch_size]

    # get patch feature vector
    fv = get_histogram(window_magnitudes, window_orientations, patch_size, num_local_cells, num_bins)

    # Take squareroot of feature vector -> normalize
    sqrt_fv = np.sqrt(fv)
    norm_fv = sqrt_fv/np.linalg.norm(sqrt_fv)

    fv = np.reshape(norm_fv, (num_local_cells**2 * num_bins, 1))
    
    return fv

def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    patch_size: int = 16,
    size_grouped_cells: int = 4,
    num_bins: int = 8
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    feat_dim = (size_grouped_cells**2) * num_bins
    fvs = np.zeros((len(X), feat_dim))

    Ix, Iy = compute_image_gradients(image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    for i in range(len(X)):
        c = X[i]
        r = Y[i]
        fvs[i] = np.squeeze(get_feat_vec(c, r, magnitudes, orientations, patch_size, size_grouped_cells, num_bins), axis=1)

    return fvs
