import os, logging, cv2, numpy as np
from flask import Flask, render_template, request, send_from_directory
from utils.analyzer import analyze_frame

# --- Flask Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- Home Page ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Upload and Predict Route ---
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read image
    frame = cv2.imread(filepath)
    
    if frame is None:
        return render_template('index.html', error="Invalid image file")

    # Run disease analysis (NEW FORMAT with multiple diseases)
    result = analyze_frame(frame)
    
    # Handle the new format with multiple diseases
    diseases = result.get("diseases", [])
    health_score = result.get("health_score", 100)
    recommendations = result.get("recommendations", [])
    
    # Get primary disease (first in list)
    primary_disease = diseases[0]["name"] if diseases else "Healthy"
    
    logging.info(f"Analyzed {file.filename}: Primary Disease={primary_disease}, Health Score={health_score}")

    return render_template(
        'index.html',
        filename=file.filename,
        disease=primary_disease,
        diseases=diseases,
        health_score=health_score,
        recommendations=recommendations,
        emotion=result.get("emotion"),
        age=result.get("age"),
        race=result.get("race")
    )

# --- Serve Uploaded Image ---
@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Run the App ---
if __name__ == '__main__':
    print("="*60)
    print("✅ Starting AI Health Analyzer (Image Upload Version)...")
    print("="*60)
    print("🌐 Local URL:    http://127.0.0.1:5000")
    print("🌐 Network URL:  http://localhost:5000")
    print("="*60)
    print("📌 Press CTRL+C to quit")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)