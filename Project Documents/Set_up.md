### 1\. `utils.py`

**Changes:** Updated the transformation functions to strictly match the logic used in your training dataset generation (e.g., using `YCrCb` for histogram equalization instead of `YUV`, and handling Alpha channels).

---

### 2\. `main.py`

1.  Orchestrates the transformations locally using `utils`.
2.  Sends transformed images to specific ports.
3.  **Voting Logic:** Aggregates results, counts votes, and returns the winner.

### 3\. `model_service.py`

You run this script 5 times in different terminals with the correct arguments.

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
