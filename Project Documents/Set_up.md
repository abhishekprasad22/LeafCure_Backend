Here are the updated files. I have synchronized the transformation logic in `utils.py` to exactly match your `create_transformed_datasets.py` script (to ensure the models see exactly what they were trained on) and implemented the "Maximum Voting" (Ensemble) logic in `main.py`.

### 1\. `utils.py`

**Changes:** Updated the transformation functions to strictly match the logic used in your training dataset generation (e.g., using `YCrCb` for histogram equalization instead of `YUV`, and handling Alpha channels).

```python
import cv2
import numpy as np
import io

# Map of supported colormap names to cv2 constants (Matching your dataset script)
_COLORMAP_MAP = {
    "JET": cv2.COLORMAP_JET,
    # Add others if you trained on them, but JET is default in your script
}

def read_image_from_bytes(file_bytes):
    """Converts uploaded bytes to an OpenCV image, handling Alpha channels."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image bytes")

    # If image has alpha channel (4 channels), drop it like in your dataset script
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Ensure it is 3 channel BGR if it came in as grayscale for consistency
    if img.ndim == 2:
         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def encode_image_to_bytes(img):
    """Converts OpenCV image back to bytes to send via HTTP."""
    _, encoded_img = cv2.imencode('.png', img)
    return io.BytesIO(encoded_img.tobytes())

# --- Transformation Functions (Synced with create_transformed_datasets.py) ---

def to_grayscale(img_bgr):
    """Converts BGR to Grayscale."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def negative_image(img):
    """Inverts the image colors."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return 255 - img

def histogram_equalize_color(img_bgr):
    """
    Equalize luminance channel (Y) in YCrCb to preserve colors.
    Matches logic from create_transformed_datasets.py
    """
    if img_bgr.ndim == 2:
        return cv2.equalizeHist(img_bgr)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def false_color_map(gray_img, colormap_name="JET"):
    """Applies a colormap. Input should ideally be grayscale."""
    # Ensure input is grayscale before mapping, as per dataset script logic
    if gray_img.ndim == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cmap = _COLORMAP_MAP.get(colormap_name.upper(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, cmap)
    return colored
```

---

### 2\. `main.py`

**Changes:**

1.  Orchestrates the transformations locally using `utils`.
2.  Sends transformed images to specific ports.
3.  **Voting Logic:** Aggregates results, counts votes, and returns the winner.

<!-- end list -->

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
import asyncio
from collections import Counter
from utils import read_image_from_bytes, encode_image_to_bytes, to_grayscale, negative_image, histogram_equalize_color, false_color_map

app = FastAPI(title="Ensemble Tea Leaf Disease Backend")

# Configuration of your microservices
# Ensure you run model_service.py on these specific ports with the correct models loaded
SERVICES = {
    "original":    {"url": "http://127.0.0.1:8001/predict", "transform": None},
    "grayscale":   {"url": "http://127.0.0.1:8002/predict", "transform": to_grayscale},
    "negative":    {"url": "http://127.0.0.1:8003/predict", "transform": negative_image},
    "false_color": {"url": "http://127.0.0.1:8004/predict", "transform": false_color_map},
    "histogram":   {"url": "http://127.0.0.1:8005/predict", "transform": histogram_equalize_color},
}

def calculate_voting_result(predictions):
    """
    Performs Maximum Voting (Majority Rule).
    If there is a tie in votes, the class with the highest cumulative confidence wins.
    """
    valid_predictions = [p for p in predictions if "error" not in p]

    if not valid_predictions:
        return {"error": "No successful predictions from microservices"}

    # 1. Count votes
    votes = Counter(p['prediction'] for p in valid_predictions)

    # 2. Find winner
    # most_common returns a list of tuples, e.g., [('healthy', 3), ('algal_spot', 2)]
    top_prediction, top_vote_count = votes.most_common(1)[0]

    # 3. Calculate Average Confidence for the winner
    # We only average the confidence of the models that voted for the winner
    winning_confidences = [p['confidence'] for p in valid_predictions if p['prediction'] == top_prediction]
    avg_confidence = sum(winning_confidences) / len(winning_confidences)

    return {
        "final_prediction": top_prediction,
        "final_confidence": round(avg_confidence, 2),
        "vote_count": top_vote_count,
        "total_models": len(valid_predictions),
        "details": valid_predictions # Send back details if frontend wants to show them
    }

async def query_microservice(client, service_name, service_info, img_cv2):
    """Helper function to transform image and send to specific microservice"""
    try:
        # 1. Apply specific transformation locally
        if service_info["transform"]:
            # Note: We copy to avoid modifying the original image for other tasks
            processed_img = service_info["transform"](img_cv2.copy())
        else:
            processed_img = img_cv2

        # 2. Re-encode to bytes to send over HTTP
        transformed_bytes_io = encode_image_to_bytes(processed_img)
        files = {'file': ('image.png', transformed_bytes_io, 'image/png')}

        # 3. Send to Microservice
        response = await client.post(service_info["url"], files=files)
        response.raise_for_status()
        res_json = response.json()

        # Add metadata for the orchestrator
        res_json['used_transform'] = service_name
        return res_json

    except Exception as e:
        print(f"Error querying {service_name}: {str(e)}")
        return {"model_name": service_name, "error": str(e)}

@app.post("/analyze_leaf")
async def analyze_leaf(file: UploadFile = File(...)):
    # Read original bytes once
    original_bytes = await file.read()

    # Decode to OpenCV format once for processing
    try:
        original_cv2 = read_image_from_bytes(original_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Parallel Request Processing
    async with httpx.AsyncClient() as client:
        tasks = []
        for name, info in SERVICES.items():
            tasks.append(query_microservice(client, name, info, original_cv2))

        # Wait for all to finish
        microservice_responses = await asyncio.gather(*tasks)

    # Calculate Final Result
    final_result = calculate_voting_result(microservice_responses)

    return final_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

### 3\. `model_service.py`

_No major logic changes required here_, but ensure you run this script 5 times in different terminals with the correct arguments.

**Example commands to run your fleet:**

```bash
# Terminal 1 (Original)
python model_service.py --model_path models/original_model.keras --port 8001 --name Original

# Terminal 2 (Grayscale)
python model_service.py --model_path models/grayscale_model.keras --port 8002 --name Grayscale

# Terminal 3 (Negative)
python model_service.py --model_path models/negative_model.keras --port 8003 --name Negative

# Terminal 4 (False Color)
python model_service.py --model_path models/false_color_model.keras --port 8004 --name FalseColor

# Terminal 5 (Histogram)
python model_service.py --model_path models/histogram_model.keras --port 8005 --name Histogram
```

### 4\. `requirements.txt`

Updated to include necessary libraries.

```text
fastapi
uvicorn
tensorflow
opencv-python
python-multipart
httpx
numpy
pillow
```

### How it works now:

1.  User sends **1 image** to `main.py` (port 8000).
2.  `main.py` converts that image into OpenCV format.
3.  `main.py` creates 4 copies and applies the math from `utils.py` (e.g., Histogram Equalization).
4.  `main.py` sends the specific version of the image to the specific model running on ports 8001-8005.
5.  The models predict and return the class.
6.  `main.py` counts the votes (e.g., "Healthy": 3 votes, "Algal Spot": 2 votes).
7.  `main.py` returns "Healthy" to the user.
