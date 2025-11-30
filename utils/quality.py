import cv2

def check_quality(frame, bbox, cfg, blur_score):
    h,w = frame.shape[:2]
    if bbox is None:
        return False, "No face"
    (x,y,x2,y2) = bbox
    face_area = max(0, (x2-x)) * max(0, (y2-y))
    if face_area / (w*h) < cfg["preprocess"]["min_face_ratio"]:
        return False, "Face too small"
    if blur_score < cfg["preprocess"]["blur_min_laplacian"]:
        return False, "Too blurry"
    # overall lighting
    L_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,0].mean()
    if L_mean < cfg["preprocess"]["min_light_L_mean"]:
        return False, "Too dark"
    return True, "OK"
