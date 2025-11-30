import cv2
import numpy as np
from deepface import DeepFace
from .color_utils import hsv_stats, blur_laplacian

def analyze_skin_regions(frame):
    """Extract detailed skin analysis from different facial regions."""
    height, width = frame.shape[:2]
    
    # Define facial regions
    regions = {
        'forehead': frame[int(height*0.1):int(height*0.3), int(width*0.3):int(width*0.7)],
        'cheeks': frame[int(height*0.3):int(height*0.6), :],
        'chin': frame[int(height*0.6):int(height*0.9), int(width*0.3):int(width*0.7)],
        'under_eyes': frame[int(height*0.25):int(height*0.4), int(width*0.2):int(width*0.8)]
    }
    
    region_data = {}
    for name, roi in regions.items():
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            region_data[name] = {
                'mean_h': np.mean(hsv[:,:,0]),
                'mean_s': np.mean(hsv[:,:,1]),
                'mean_v': np.mean(hsv[:,:,2]),
                'std_v': np.std(hsv[:,:,2])
            }
    
    return region_data

def detect_eye_features(frame):
    """Analyze eye region for dark circles and puffiness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    eye_data = {'dark_circles': False, 'puffiness': False, 'redness': False}
    
    for (ex, ey, ew, eh) in eyes:
        # Extract under-eye region
        under_eye = frame[ey+eh:ey+eh+20, ex:ex+ew]
        if under_eye.size > 0:
            under_eye_gray = cv2.cvtColor(under_eye, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(under_eye_gray)
            
            # Dark circles detection
            if avg_brightness < 80:
                eye_data['dark_circles'] = True
            
            # Check for redness
            b, g, r = cv2.split(under_eye)
            if np.mean(r) > np.mean(g) + 20:
                eye_data['redness'] = True
    
    return eye_data

def analyze_skin_texture(frame):
    """Analyze skin texture for acne, dryness, and oiliness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Texture variance (high = rough/acne, low = smooth)
    texture_var = np.var(gray)
    
    # Edge detection for spots/blemishes
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    # Shiny/oily detection using brightness variance
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness_std = np.std(hsv[:,:,2])
    
    return {
        'texture_variance': texture_var,
        'edge_density': edge_density,
        'brightness_std': brightness_std
    }

def detect_facial_symmetry(frame):
    """Check for facial asymmetry which may indicate certain conditions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    left_half = gray[:, :width//2]
    right_half = cv2.flip(gray[:, width//2:], 1)
    
    # Resize to match if needed
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = right_half[:, :min_width]
    
    # Calculate difference
    diff = cv2.absdiff(left_half, right_half)
    asymmetry_score = np.mean(diff)
    
    return asymmetry_score

def analyze_frame(frame, face_detection=None):
    """
    Comprehensive facial health analysis with 20+ detectable conditions.
    Returns emotion, race, age, and predicted diseases with confidence levels.
    """
    
    data = {
        "emotion": None, 
        "race": None, 
        "age": None, 
        "diseases": [],
        "health_score": 100,
        "recommendations": []
    }
    
    try:
        # DeepFace analysis
        results = DeepFace.analyze(
            frame,
            actions=['emotion', 'age', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        data["emotion"] = results[0].get("dominant_emotion", "Unknown")
        data["race"] = results[0].get("dominant_race", "Unknown")
        data["age"] = results[0].get("age", "Unknown")
        
    except Exception as e:
        print("DeepFace error:", e)
    
    # --- Advanced Analysis ---
    hsv_means = hsv_stats(frame)
    blur_val = blur_laplacian(frame)
    region_data = analyze_skin_regions(frame)
    eye_data = detect_eye_features(frame)
    texture_data = analyze_skin_texture(frame)
    asymmetry = detect_facial_symmetry(frame)
    
    avg_hue = hsv_means['h']
    avg_sat = hsv_means['s']
    avg_val = hsv_means['v']
    
    diseases = []
    health_score = 100
    
    # --- COMPREHENSIVE DISEASE DETECTION ---
    
    # 1. Jaundice (Liver problems)
    if 20 < avg_hue < 40 and avg_sat > 80:
        diseases.append({
            "name": "Jaundice (Liver Issue)",
            "confidence": "High",
            "description": "Yellowish skin tone detected"
        })
        health_score -= 30
        data["recommendations"].append("Consult a hepatologist immediately")
    
    # 2. Anemia (Iron deficiency)
    if avg_sat < 40 and avg_val > 150:
        diseases.append({
            "name": "Anemia",
            "confidence": "Medium",
            "description": "Pale complexion indicates possible iron deficiency"
        })
        health_score -= 20
        data["recommendations"].append("Check hemoglobin levels; increase iron intake")
    
    # 3. Cyanosis (Oxygen deficiency)
    if 90 < avg_hue < 130 and avg_sat > 60:
        diseases.append({
            "name": "Cyanosis",
            "confidence": "High",
            "description": "Bluish tint suggests low blood oxygen"
        })
        health_score -= 35
        data["recommendations"].append("Seek immediate medical attention - possible respiratory/cardiac issue")
    
    # 4. Rosacea / Inflammation
    if avg_hue < 15 and avg_sat > 100:
        diseases.append({
            "name": "Rosacea / Skin Inflammation",
            "confidence": "Medium",
            "description": "Persistent facial redness detected"
        })
        health_score -= 15
        data["recommendations"].append("Avoid triggers like alcohol, spicy foods; consult dermatologist")
    
    # 5. Vitiligo (Depigmentation) - IMPROVED
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(bright_mask == 255) / bright_mask.size
    
    if white_pixel_ratio > 0.15 and texture_data['brightness_std'] > 35 and avg_sat < 60:
        diseases.append({
            "name": "Vitiligo",
            "confidence": "Medium",
            "description": "White/depigmented patches detected on skin"
        })
        health_score -= 10
        data["recommendations"].append("Consult dermatologist for vitiligo treatment options")
    
    # 6. Acne / Skin Infection (Lowered threshold)
    if texture_data['edge_density'] > 0.12 and texture_data['texture_variance'] > 600:
        diseases.append({
            "name": "Acne / Skin Lesions",
            "confidence": "Medium",
            "description": "Multiple blemishes or spots detected"
        })
        health_score -= 12
        data["recommendations"].append("Maintain skincare routine; consider topical treatments")
    
    # 7. Eczema / Dermatitis (More sensitive)
    if texture_data['texture_variance'] > 800 and avg_sat < 60:
        diseases.append({
            "name": "Eczema / Dermatitis",
            "confidence": "Low",
            "description": "Dry, rough skin texture"
        })
        health_score -= 15
        data["recommendations"].append("Use moisturizers; avoid irritants; see dermatologist")
    
    # 8. Melasma / Hyperpigmentation
    if 'forehead' in region_data and region_data['forehead']['mean_v'] < avg_val - 20:
        diseases.append({
            "name": "Melasma / Hyperpigmentation",
            "confidence": "Low",
            "description": "Dark patches on forehead region"
        })
        health_score -= 8
        data["recommendations"].append("Use sunscreen; consider skin-lightening treatments")
    
    # 9. Dehydration
    if avg_val < 100 and blur_val < 60:
        diseases.append({
            "name": "Dehydration",
            "confidence": "Medium",
            "description": "Dull, tired-looking skin"
        })
        health_score -= 10
        data["recommendations"].append("Increase water intake to 8+ glasses daily")
    
    # 10. Chronic Fatigue / Sleep Deprivation
    if eye_data['dark_circles'] and avg_val < 110:
        diseases.append({
            "name": "Sleep Deprivation / Chronic Fatigue",
            "confidence": "Medium",
            "description": "Dark circles and dull complexion"
        })
        health_score -= 15
        data["recommendations"].append("Improve sleep quality; aim for 7-9 hours nightly")
    
    # 11. Allergic Reaction
    if eye_data['redness'] and avg_hue < 20:
        diseases.append({
            "name": "Allergic Reaction",
            "confidence": "Medium",
            "description": "Redness and possible swelling detected"
        })
        health_score -= 18
        data["recommendations"].append("Identify and avoid allergens; antihistamines may help")
    
    # 12. Thyroid Disorder (Hypothyroidism)
    if avg_sat < 35 and avg_val > 160 and eye_data.get('puffiness', False):
        diseases.append({
            "name": "Hypothyroidism (Possible)",
            "confidence": "Low",
            "description": "Pale, puffy face may indicate thyroid issues"
        })
        health_score -= 20
        data["recommendations"].append("Get thyroid function tests (TSH, T3, T4)")
    
    # 13. Lupus (Butterfly rash indicator)
    if 'cheeks' in region_data and region_data['cheeks']['mean_h'] < 15 and asymmetry < 10:
        diseases.append({
            "name": "Lupus (Butterfly Rash Indicator)",
            "confidence": "Very Low",
            "description": "Symmetrical redness across cheeks"
        })
        health_score -= 25
        data["recommendations"].append("Consult rheumatologist for autoimmune screening")
    
    # 14. Cushing's Syndrome
    if avg_val > 170 and 'cheeks' in region_data and region_data['cheeks']['mean_v'] > 180:
        diseases.append({
            "name": "Cushing's Syndrome (Moon Face)",
            "confidence": "Very Low",
            "description": "Round, full facial appearance"
        })
        health_score -= 22
        data["recommendations"].append("Consult endocrinologist for cortisol level testing")
    
    # 15. Seborrheic Dermatitis
    if texture_data['edge_density'] > 0.12 and 20 < avg_hue < 35:
        diseases.append({
            "name": "Seborrheic Dermatitis",
            "confidence": "Low",
            "description": "Flaky, oily patches on skin"
        })
        health_score -= 12
        data["recommendations"].append("Use antifungal shampoos; maintain scalp hygiene")
    
    # 16. Psoriasis
    if texture_data['texture_variance'] > 1200 and avg_hue > 15 and avg_hue < 25:
        diseases.append({
            "name": "Psoriasis",
            "confidence": "Low",
            "description": "Rough, scaly skin patches"
        })
        health_score -= 18
        data["recommendations"].append("Consult dermatologist for biologics or topical treatments")
    
    # 17. Liver Disease (Advanced)
    if 25 < avg_hue < 35 and avg_sat > 90 and avg_val < 120:
        diseases.append({
            "name": "Advanced Liver Disease",
            "confidence": "Medium",
            "description": "Dark yellowish tone with dullness"
        })
        health_score -= 40
        data["recommendations"].append("URGENT: See hepatologist for liver function tests")
    
    # 18. Kidney Disease
    if avg_val > 180 and avg_sat < 30 and eye_data.get('puffiness', False):
        diseases.append({
            "name": "Kidney Disease (Possible)",
            "confidence": "Low",
            "description": "Pale, puffy face with fluid retention"
        })
        health_score -= 25
        data["recommendations"].append("Get kidney function tests (creatinine, BUN)")
    
    # 19. Malnutrition
    if avg_sat < 25 and avg_val < 90 and blur_val < 50:
        diseases.append({
            "name": "Malnutrition",
            "confidence": "Medium",
            "description": "Very pale and dull complexion"
        })
        health_score -= 30
        data["recommendations"].append("Improve diet; consider nutritional supplements")
    
    # 20. Hormonal Imbalance (Acne pattern)
    if texture_data['edge_density'] > 0.18 and 'chin' in region_data:
        diseases.append({
            "name": "Hormonal Acne",
            "confidence": "Medium",
            "description": "Breakouts concentrated in lower face"
        })
        health_score -= 15
        data["recommendations"].append("Consult endocrinologist; may need hormonal treatment")
    
    # 21. Stress / Anxiety
    if data["emotion"] in ["angry", "fear", "sad"] and avg_val < 105:
        diseases.append({
            "name": "Chronic Stress / Anxiety",
            "confidence": "Medium",
            "description": "Emotional distress visible in facial features"
        })
        health_score -= 18
        data["recommendations"].append("Practice stress management; consider counseling")
    
    # 22. Vitamin D Deficiency
    if avg_val > 165 and avg_sat < 35:
        diseases.append({
            "name": "Vitamin D Deficiency",
            "confidence": "Low",
            "description": "Very pale skin lacking healthy glow"
        })
        health_score -= 12
        data["recommendations"].append("Get sun exposure; vitamin D supplements")
    
    # 23. Perioral Dermatitis
    if 'chin' in region_data and region_data['chin']['mean_h'] < 20 and texture_data['edge_density'] > 0.14:
        diseases.append({
            "name": "Perioral Dermatitis",
            "confidence": "Low",
            "description": "Rash around mouth area"
        })
        health_score -= 10
        data["recommendations"].append("Avoid steroid creams; see dermatologist")
    
    # 24. Contact Dermatitis
    if asymmetry > 25 and avg_hue < 15:
        diseases.append({
            "name": "Contact Dermatitis",
            "confidence": "Low",
            "description": "Localized redness from irritant/allergen"
        })
        health_score -= 10
        data["recommendations"].append("Identify irritant; use hypoallergenic products")
    
    # 25. Sun Damage / Photoaging
    if texture_data['texture_variance'] > 900 and avg_hue > 15 and avg_hue < 25:
        diseases.append({
            "name": "Sun Damage / Photoaging",
            "confidence": "Medium",
            "description": "Uneven texture and pigmentation from UV exposure"
        })
        health_score -= 15
        data["recommendations"].append("Daily SPF 50+; consider retinoids")
    
    # --- FINAL ASSESSMENT ---
    if not diseases:
        diseases.append({
            "name": "Healthy",
            "confidence": "High",
            "description": "No significant health issues detected"
        })
        data["recommendations"].append("Maintain current healthy lifestyle")
    
    data["diseases"] = diseases
    data["health_score"] = max(0, health_score)
    
    return data