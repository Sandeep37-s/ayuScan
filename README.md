🩺 AI Health Analyzer – Facial Disease Detection Using Computer Vision
📘 Overview

AI Health Analyzer is an intelligent web application built using Flask and OpenCV that predicts visible human diseases from facial images.
It analyzes skin tone, color, and eye region to detect early symptoms of conditions such as:

🟡 Jaundice (yellowish skin/eyes)

⚪ Anemia (pale skin)

🔵 Cyanosis (bluish tint)

⚫ Fatigue / Stress (dark dull face tone)

🔴 Allergies / Inflammation (reddish skin)

The system allows users to upload an image and instantly receive a health analysis report based on facial features.

🚀 Features

✅ Upload any clear human face photo (JPG/PNG)
✅ Detects visible health symptoms using color and brightness metrics
✅ Real-time skin-tone and brightness analysis using HSV color model
✅ Simple, fast Flask web interface
✅ Extensible for deep learning models (CNN / Eye analysis / DeepFace integration)

🧠 Tech Stack
Category	Tools / Libraries
Backend Framework	Flask
Computer Vision	OpenCV
Image Analysis	NumPy, HSV color statistics
ML/AI Logic	Custom rule-based analyzer
Frontend	HTML5, CSS3 (responsive)
Logging	Python logging module
🧩 Project Structure
AI-Health-Analyzer/
│
├── app.py                  # Main Flask app
├── templates/
│   └── index.html          # Frontend page for uploading and viewing results
├── utils/
│   ├── analyzer.py         # Core disease detection logic
│   └── color_utils.py      # Helper for HSV and tone analysis
├── uploads/                # Uploaded user images
├── logs/                   # Application logs
└── README.md               # This file

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/AI-Health-Analyzer.git
cd AI-Health-Analyzer

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # for Windows
# or
source venv/bin/activate   # for Linux/macOS

3️⃣ Install Requirements
pip install -r requirements.txt


Example requirements.txt

flask
opencv-python
numpy


(If you’re using DeepFace or Mediapipe, add them too.)

4️⃣ Run the App
python app.py


Then open your browser and go to:

http://127.0.0.1:5000

🖼️ Usage

Click Choose File and upload a face image.

Click Analyze.

The model will process your image and display:

The uploaded photo

The predicted visible disease

🧬 Example Output
Uploaded Image	Predicted Disease

	🟡 Possible Jaundice

	⚪ Possible Anemia
📈 Future Improvements

Add eye region detection for better jaundice/fatigue analysis

Integrate DeepFace or CNN-based classification for precise detection

Real-time webcam scanning using MediaPipe

Deploy on Render / HuggingFace / Streamlit Cloud

👨‍💻 Author

Sandeep Kumar
B.Tech CSE — Central University of Jammu
Passionate about AI, Computer Vision, and HealthTech innovations.

📧 Email: sandeepkumar6200187840@gmail.com
]
🌐 GitHub: https://github.com/Sandeep37-s

🪪 License

This project is licensed under the MIT License – you’re free to use, modify, and share with attribution.

⭐ Show Support

If you like this project, please ⭐ the repo to support development!


