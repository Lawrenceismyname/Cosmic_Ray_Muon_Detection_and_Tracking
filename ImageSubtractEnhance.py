import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def enhance_differences_scalar(img1, img2, enhancement_factor=2.0):
    """
    Subtract images and multiply by scalar to enhance differences
    """

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Subtract images
    difference = cv2.absdiff(img1, img2)
    
    # Convert to float for multiplication
    difference_float = difference.astype(np.float32)
    
    # Multiply by enhancement factor
    enhanced_difference = difference_float * enhancement_factor
    
    # Clip to valid range and convert back
    enhanced_difference = np.clip(enhanced_difference, 0, 255).astype(np.uint8)
    
    return difference, enhanced_difference


start = 86400
end = 86421

for i in range(start, end):
    j = i + 1
    png_path1 = f"Timeline 1_000{i:05d}.png"
    png_path2 = f"Timeline 1_000{j:05d}.png"
    img1 = cv2.imread(png_path1)
    img2 = cv2.imread(png_path2)

    subtracted, enhanced_subtracted = enhance_differences_scalar(img1, img2, 10)

    plt.imshow(enhanced_subtracted)
    plt.show()


