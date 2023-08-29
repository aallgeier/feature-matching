import numpy as np

def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    dists = np.zeros((len(features1), len(features2)))

    for i, feat1 in enumerate(features1):
        dists[i] = np.sqrt(np.sum((features2 - feat1)**2, axis=1))

    return dists


def match_features_ratio_test(features1,features2):
    """ Nearest-neighbor distance ratio feature matching.

    distance ratio matching:
        Let t_a be a feature in im1. Let d1, d2 be the closest and second closest 
        features in im2 to t_a. If d1 is much smaller than d2, then d1 is
        likely a match. However, if d1 and d2 are similar, there is a higher
        chance that we don't have a correct match. 

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    threshold = 0.88
    dists_mtrx = compute_feature_distances(features1, features2)

    matches = []
    confidences = []

    # features1 --> features2 matches
    for i, dists_from_feat_1 in enumerate(dists_mtrx):
    
        # Sort the indicies of dists_from_feat1 with entries in ascending order.
        sorted_dist_ind = np.argsort(dists_from_feat_1)
        ind_min1 = sorted_dist_ind[0]
        ind_min2 = sorted_dist_ind[1]

        # Get min distance and second min distance.
        N1 = dists_from_feat_1[ind_min1]
        N2 = dists_from_feat_1[ind_min2]

        # Compute ratio.
        ratio = N1/N2

        # If ratio is smaller than threshold, it's a match.
        if ratio < threshold:
            match = [i, ind_min1]
            if match not in matches:
                matches.append(match)
                confidences.append(ratio)

    matches = np.reshape(np.array(matches), (-1, 2))
    confidences = np.reshape(np.array(confidences), (-1, 1))
    
    return matches, confidences
