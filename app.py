import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas

from inference_model import infer_image  # Import the separated Roboflow model logic

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key for production

# === App Configuration ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BLUR_THRESHOLD = 50  # Image sharpness threshold

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Helper Functions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('home'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            if image is None:
                return render_template('index.html', error="Invalid image file. Please upload a valid image.")

            blur_variance = check_blur(image)
            if blur_variance < BLUR_THRESHOLD:
                return render_template('index.html', error="Image is too blurry. Please upload a clearer image.")

            # === Roboflow Prediction ===
            try:
                result_data = infer_image(filepath)
                result = result_data["prediction"]
                confidence = result_data["confidence"]

                session['prediction'] = result
                session['confidence'] = confidence
                session['sharpness'] = f"{blur_variance:.1f}"

                return render_template('result.html',
                                       prediction=result,
                                       confidence=confidence,
                                       sharpness=session['sharpness'])

            except Exception as e:
                print(f"[ERROR] Roboflow inference error: {e}")
                return render_template('index.html', error="Model inference failed. Please try again.")

        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            return render_template('index.html', error="An error occurred while processing the image.")
    
    return redirect(url_for('home'))

@app.route('/download-report')
def download_report():
    if 'prediction' not in session:
        return redirect(url_for('home'))

    try:
        buffer = BytesIO()
        p = canvas.Canvas(buffer)

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 800, "Skin Cancer Detection Report")

        p.setFont("Helvetica", 12)
        y = 750
        for key, value in {
            "Prediction": session.get('prediction', 'N/A'),
            "Confidence": session.get('confidence', 'N/A'),
            "Image Sharpness": session.get('sharpness', 'N/A')
        }.items():
            p.drawString(100, y, f"{key}: {value}")
            y -= 30

        p.save()
        buffer.seek(0)

        return send_file(buffer,
                         mimetype='application/pdf',
                         as_attachment=True,
                         download_name='skin_cancer_report.pdf')
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return redirect(url_for('home'))

# === Run Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
