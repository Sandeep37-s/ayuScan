import cv2, numpy as np, mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Key landmark indices (MediaPipe Face Mesh)
L_EYE = [33,133,159,145,153,154,155]         # approx left eye rim
R_EYE = [263,362,386,374,380,381,382]
UNDER_EYE_L = [145,153,154,155,157]          # lower lid region
UNDER_EYE_R = [374,380,381,382,384]
FOREHEAD = [10,338,297,332,284,251,389,356,454]  # rough polygon
L_CHEEK = [50,101,118,229,230,205]
R_CHEEK = [280,330,347,449,448,425]

def _landmark_pts(landmarks, w, h, idxs):
    return np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idxs], dtype=np.int32)

def polygon_crop(img, pts, pad=0):
    x,y,w,h = cv2.boundingRect(pts)
    x0=max(x-pad,0); y0=max(y-pad,0)
    x1=min(x+w+pad, img.shape[1]); y1=min(y+h+pad, img.shape[0])
    roi = img[y0:y1, x0:x1].copy()
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    pts_shifted = pts - [x0,y0]
    cv2.fillPoly(mask, [pts_shifted], 255)
    return cv2.bitwise_and(roi, roi, mask=mask)

def extract_rois(frame, face_landmarks):
    h, w = frame.shape[:2]
    l_eye = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, L_EYE))
    r_eye = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, R_EYE))
    under_l = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, UNDER_EYE_L))
    under_r = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, UNDER_EYE_R))
    forehead = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, FOREHEAD))
    l_cheek = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, L_CHEEK))
    r_cheek = polygon_crop(frame, _landmark_pts(face_landmarks, w, h, R_CHEEK))
    return {
        "l_eye": l_eye, "r_eye": r_eye,
        "under_l": under_l, "under_r": under_r,
        "forehead": forehead, "l_cheek": l_cheek, "r_cheek": r_cheek
    }
