# Three things everyone should know to improve object retrieval
# Relja Arandjelovi, Andrew Zisserman

def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # Use Pythagorean to get magnitudes. 
    magnitudes = np.sqrt(Ix**2 + Iy**2)

    # Get tangent values at each location and use arctan appropriately. 
    orientations = np.arctan2(Iy, Ix)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations