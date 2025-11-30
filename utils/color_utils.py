import cv2
import numpy as np

def hsv_stats(image):
    """Returns mean HSV values of the given image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    return {'h': h_mean, 's': s_mean, 'v': v_mean}

def blur_laplacian(image):
    """Returns a measure of blurriness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    val = cv2.Laplacian(gray, cv2.CV_64F).var()
    return val
