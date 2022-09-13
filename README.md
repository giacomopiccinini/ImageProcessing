# ImageProcessing

Here I collect useful tools and algorithms to process images without resorting to Deep Learning. Whilst DL is certainly superior in general, an effective pre-processing is extremely useful and should not be underestimated. 

## MaskNN

Here I implemented a simple k-Nearest Neighbour algorithm in the case of masks using pure NumPy. By mask I mean an image composed of 0's and 1's only. The region of interest (ROI) is that indicated by 1's. 

Given two masks, mask_1 and mask_2, for each point in mask_1 the algorithm returns the coordinates of the k elements of mask_2 which are closest to it, using a Manhattan distance. Crucially, this algorithm is optimised, as "for" loops are avoided and NumPy broadcasting preferred. 

## MaskContour

This algorithm retrieves the contour (boundary) of a ROI using jump in gradients. 

## EdgeDetection

This algorithm is useful for detecting edges in images. It is definitely not my original work, but it is nonetheless useful to have it written down somewhere. Notice: the Gaussian blur pre-processing is a recommended step for removing noise and thus improving the overall efficiency. Thresholds for canny edge detection should be set depending on the case at hand. 

## Segmentation

A simple re-writing of the Otsu-Voronoi segmentation algorithm as implemented in pyclesperanto. Once again, work is not original by any means and it is to be considered as a reminder. 