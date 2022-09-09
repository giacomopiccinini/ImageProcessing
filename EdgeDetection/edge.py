import cv2

# Read image
image = cv2.imread("image.png", -1)

# Convert to 8-bit (necessary if e.g. image is 16-bit)
image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Blur image to remove noise (useful preprocessing)
blurred = cv2.GaussianBlur(image_8bit, (3,3), 0)

# Retrieve edges (adjust threshold based on your use case!)
edges = cv2.Canny(image=blurred, threshold1=100, threshold2=10)

# Plot result
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
