import cv2, numpy as np

def tint_region(base_img, region_img, color_bgr, alpha=0.35):
    overlay = base_img.copy()
    mask = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    color = np.full_like(base_img, color_bgr, dtype=np.uint8)
    color = cv2.bitwise_and(color, color, mask=mask)
    cv2.addWeighted(color, alpha, overlay, 1-alpha, 0, overlay)
    return overlay
