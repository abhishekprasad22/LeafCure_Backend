import argparse
import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pathlib

# 1. Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the .keras file")
parser.add_argument("--port", type=int, required=True, help="Port to run this service on")
parser.add_argument("--name", type=str, required=True, help="Name of this model service")
args = parser.parse_args()

# --- PATH FIXER START ---
# This block ensures the path works on Windows with spaces/slashes
raw_path = args.model_path
# Convert to absolute path object
path_obj = pathlib.Path(raw_path).resolve()
# Convert to string with correct OS separators (e.g., backslashes on Windows)
fixed_model_path = str(path_obj)

print(f"[-] Raw Path Input:   {raw_path}")
print(f"[-] Fixed Absolute:   {fixed_model_path}")

if not path_obj.exists():
    print(f"CRITICAL ERROR: Python still cannot see the file at: {fixed_model_path}")
    # Check if it's a relative path issue
    print(f"    (Current Working Directory is: {os.getcwd()})")
    exit(1)
# --- PATH FIXER END ---

# 2. Initialize App and Load Model
app = FastAPI(title=f"{args.name} Service")

print(f"Loading model from {fixed_model_path}...")
try:
    model = tf.keras.models.load_model(fixed_model_path)
    print(f"Model {args.name} loaded successfully!")
except Exception as e:
    print(f"FAILED to load model. Error details:\n{e}")
    exit(1)

# Define class names
CLASS_NAMES = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']

def preprocess_image(image_bytes):
    """Resize and scale image for the model"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256)) 
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, 0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = preprocess_image(contents)
    
    predictions = model.predict(processed_img)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return {
        "model_name": args.name,
        "prediction": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port)