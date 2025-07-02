import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Load pre-trained model
try:
    model = tf.keras.models.load_model(r"C:\Users\rajga\SRP PROJECT\efficientnetb0.h5")
    print("Model loaded successfully.")
    print("Model output shape:", model.output_shape)  # Debug: Check output classes
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Class labels for binary classification
CLASS_NAMES = ["No Cancer (Benign)", "Cancer (Malignant)"]
BLUR_THRESHOLD = 50  # Adjust based on empirical testing
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept a prediction

def check_blur(image):
    """Check image blur using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture or 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display live feed
    cv2.imshow("Skin Cancer Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('c'):
        # Check image focus
        blur_variance = check_blur(frame)
        if blur_variance < BLUR_THRESHOLD:
            print(f"Image too blurry (Variance: {blur_variance:.1f}). Please recapture.")
            continue

        # Convert and preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame).resize((224, 224))
        image_array = img_to_array(pil_image)
        processed_image = preprocess_input(image_array)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        # Predict
        predictions = model.predict(processed_image)
        confidence = predictions[0][1]  # Confidence for+ "Cancer (Malignant)"
        predicted_class = 1 if confidence >= CONFIDENCE_THRESHOLD else 0

        # Debug: Print predictions
        print("Raw Predictions:", predictions)
        print("Confidence for Cancer:", confidence)

        # Prepare result
        result_text = (f"Prediction: {CLASS_NAMES[predicted_class]} "
                       f"({confidence*100:.2f}% confidence)")

        # Display results
        display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(display_frame, result_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Sharpness: {blur_variance:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Analysis Result", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()