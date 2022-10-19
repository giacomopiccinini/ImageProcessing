import cupy as cp

def nn(mask_1, mask_2, k):

    """ For each point in mask_1 the k nearest neighbour belonging to mask_2 are found. 
    The metric used is Manhattan."""

    # Extract coordinates of points of mask_1
    mask_1_coordinates = cp.argwhere(mask_1)
    
    # Extract coordinates of points of mask_2
    mask_2_coordinates = cp.argwhere(mask_2)

    # Compute differences along x and y
    difference = mask_1_coordinates[:, None] - mask_2_coordinates

    # Compute the moduli of difference along each direction
    moduli = cp.abs(difference)

    # Sum moduli to obtain the actual Manhattan metric
    metrics = cp.sum(moduli, axis = 2)

    # Sort points by distance
    ordered_distances = cp.argsort(metrics, axis = 1)

    # Keep only the first k neighbours and obtain indices of their coordinates
    nn_indices = ordered_distances[:, : k]

    # Retrieve actual coordinates
    nn_coordinates = mask_2_coordinates[nn_indices]

    return nn_coordinates



def nn_value(image, mask_1, mask_2, k):

    """ Retrieve alues of k nearest neighbours """

    # Find NN coordinates
    nn_coordinates = nn(mask_1, mask_2, k)

    # Retrieve NN values (syntax is convoluted)
    nn_values = image[tuple(nn_coordinates.T.tolist())].T 

    return nn_values


