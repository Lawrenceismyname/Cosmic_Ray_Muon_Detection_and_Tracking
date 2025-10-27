import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image

def subtract_images(img1, img2):
    """Subtract two images using OpenCV"""

    # Ensure same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Absolute difference
    subtracted = cv2.absdiff(img1, img2)
    return subtracted

def display_images(original1, original2, result):
    """Display original images and result"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert BGR to RGB for display
    axes[0].imshow(cv2.cvtColor(original1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(original2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Bruh")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def histogram_equalization_color(img):
    """Apply histogram equalization to color image"""

    # Convert to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply equalization to Y channel (luminance)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # Convert back to BGR
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img, equalized

def plot_histograms(original, equalized, title="Histogram Comparison"):
    """Plot histograms before and after equalization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original image and histogram
    if len(original.shape) == 2:  # Grayscale
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[1,0].hist(original.ravel(), 256, [0,256])
        axes[1,0].set_title('Original Histogram')
    else:  # Color
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image')
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([original], [i], None, [256], [0,256])
            axes[1,0].plot(hist, color=col)
        axes[1,0].set_title('Original Histogram')
    
    # Equalized image and histogram
    if len(equalized.shape) == 2:  # Grayscale
        axes[0,1].imshow(equalized, cmap='gray')
        axes[0,1].set_title('Equalized Image')
        axes[1,1].hist(equalized.ravel(), 256, [0,256])
        axes[1,1].set_title('Equalized Histogram')
    else:  # Color
        axes[0,1].imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Equalized Image')
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([equalized], [i], None, [256], [0,256])
            axes[1,1].plot(hist, color=col)
        axes[1,1].set_title('Equalized Histogram')
    
    plt.tight_layout()
    plt.show()

def adaptive_histogram_equalization_color(img):
    """Apply CLAHE to color image using LAB color space"""
    # Read color image
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to BGR
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img, equalized


png_path1 = "Timeline 1_00086400.png"
png_path2 = "Timeline 1_00086401.png"
img1 = cv2.imread(png_path1)
img2 = cv2.imread(png_path2)


subtracted = subtract_images(img1, img2)

# display_images(img1, img2, subtracted)

img, histogramed = adaptive_histogram_equalization_color(subtracted)

plot_histograms(img, histogramed)




