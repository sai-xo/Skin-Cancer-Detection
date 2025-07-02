# inference_model.py

import os
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="4FSwNf8OGqICN29AmgIQ"
)

ROBOFLOW_MODEL_ID = "skin-cancer-recogniser/2"  # ✅ Correct Model ID

def infer_image(image_path):
    try:
        response = CLIENT.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
        predictions = response.get("predictions", [])

        if not predictions:
            return {
                "prediction": "No Cancer Detected",
                "confidence": "0%"
            }

        top_prediction = predictions[0]
        class_name = top_prediction.get("class", "Unknown")
        confidence_val = top_prediction.get("confidence", 0)

        return {
            "prediction": class_name.capitalize(),
            "confidence": f"{confidence_val * 100:.2f}%"
        }

    except Exception as e:
        print(f"[ERROR] Roboflow inference failed: {e}")
        raise RuntimeError("Model inference failed.") from e
