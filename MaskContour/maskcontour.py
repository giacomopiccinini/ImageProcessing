import numpy as np

def find_contour(mask):
    
    """ Find contour of a mask """

    # Compute gradients along x and y direction
    grad_y, grad_x = np.gradient(mask)
    
    # Take their absolute value (not interested in sign)
    grad_y = np.abs(grad_y)
    grad_x = np.abs(grad_x)
        
    # The gradients will be non-zero precisely where a jump between
    # background (i.e. 0's) and ROI (i.e. 1's) is present. That is,
    # the contour of the ROI
    contour = grad_x + grad_y

    # Ensure the contour is normalised
    contour[contour != 0] = 1

    # Ensure the contour is of the right type
    contour = contour.astype('uint8')
    
    return contour