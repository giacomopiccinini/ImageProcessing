import numpy as np

def nn(mask_1, mask_2, k):

    """ For each point in mask_1 the k nearest neighbour belonging to mask_2 are found. 
    The metric used is Manhattan. """

    # Extract coordinates of points of mask_1
    mask_1_coordinates = np.argwhere(mask_1)
    
    # Extract coordinates of points of mask_2
    mask_2_coordinates = np.argwhere(mask_2)

    # Compute differences along x and y
    difference = mask_1_coordinates[:, None] - mask_2_coordinates

    # Compute the moduli of difference along each direction
    moduli = np.abs(difference)

    # Sum moduli to obtain the actual Manhattan metric
    metrics = np.sum(moduli, axis = 2)

    # Sort points by distance
    ordered_distances = np.sort(metrics, axis = 1)

    # Keep only the first k neighbours and obtain indices of their coordinates
    nn_indices = ordered_distances[:, : k]

    # Retrieve actual coordinates
    nn_coordinates = mask_2_coordinates[nn_indices]

    return nn_coordinates