import os
import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import httpx
import asyncio
from collections import Counter
from typing import Optional
from utils import read_image_from_bytes, encode_image_to_bytes, to_grayscale, negative_image, histogram_equalize_color, false_color_map, remove_background_add_white
from weather_engine import get_average_temperature, refine_prediction_by_weather, calculate_voting_result
from supabase import create_client, Client
from dotenv import load_dotenv

# 👇 1. Load the .env file immediately
load_dotenv()

# --- 🟢 SUPABASE CONFIGURATION (Phase 2) ---
# Replace these with your actual details from Phase 1
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ CRITICAL ERROR: Supabase credentials not found in .env file!")
    supabase = None
else:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase (Securely)!")
    except Exception as e:
        print(f"⚠️ Connection Failed: {e}")
        supabase = None

app = FastAPI(title="Ensemble Tea Leaf Disease Detection Backend")

# Configuration of your microservices
# Ensure you run model_service.py on these specific ports with the correct models loaded
SERVICES = {
    "original":    {"url": "http://127.0.0.1:8001/predict", "transform": None},
    "grayscale":   {"url": "http://127.0.0.1:8002/predict", "transform": to_grayscale},
    "negative":    {"url": "http://127.0.0.1:8003/predict", "transform": negative_image},
    "false_color": {"url": "http://127.0.0.1:8004/predict", "transform": false_color_map},
    "histogram":   {"url": "http://127.0.0.1:8005/predict", "transform": histogram_equalize_color},
}

# def calculate_voting_result(predictions):
#     """
#     Performs Maximum Voting (Majority Rule).
#     If there is a tie in votes, the class with the highest cumulative confidence wins.
#     """
#     valid_predictions = [p for p in predictions if "error" not in p]
    
#     if not valid_predictions:
#         return {"error": "No successful predictions from microservices"}

#     # 1. Count votes
#     votes = Counter(p['prediction'] for p in valid_predictions)
    
#     # 2. Find winner
#     # most_common returns a list of tuples, e.g., [('healthy', 3), ('algal_spot', 2)]
#     top_prediction, top_vote_count = votes.most_common(1)[0]
    
#     # 3. Calculate Average Confidence for the winner
#     # We only average the confidence of the models that voted for the winner
#     winning_confidences = [p['confidence'] for p in valid_predictions if p['prediction'] == top_prediction]
#     avg_confidence = sum(winning_confidences) / len(winning_confidences)

#     return {
#         "final_prediction": top_prediction,
#         "final_confidence": round(avg_confidence, 2),
#         "vote_count": top_vote_count,
#         "total_models": len(valid_predictions),
#         "details": valid_predictions # Send back details if frontend wants to show them
#     }

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

# --- 🟢 HISTORY SAVING FUNCTION ---
def save_prediction_to_db(user_id, image_bytes, result):
    if not supabase:
        return
        
    print(f"💾 Saving history for User: {user_id}")
    try:
        # 1. Generate a unique filename
        filename = f"{user_id}/{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
        
        # 2. Upload Image to Supabase Storage
        supabase.storage.from_("leaf_images").upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 3. Get Public URL of the image
        # (This assumes the bucket is public or you use signed URLs. 
        # For simplicity in this app, we store the path)
        image_path = filename 
        
        # 4. Insert Record into DB
        data = {
            "user_id": user_id,
            "image_path": image_path,
            "prediction": result['final_prediction'],
            "confidence": result['final_confidence']
        }
        supabase.table("predictions").insert(data).execute()
        print("✅ History saved successfully!")
        
    except Exception as e:
        print(f"❌ Failed to save history: {e}")
# ----------------------------------

@app.post("/analyze_leaf")
async def analyze_leaf(
    file: UploadFile = File(...),
    use_weather: bool = Form(False),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    user_id: Optional[str] = Form(None) # 👈 NEW PARAMETER
):
    
    print(f"Received file: {file.filename}")
    print(f"Weather Logic Enabled: {use_weather}")

    print(f"Content-Type: {file.content_type}")
    print(f"File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    # Read original bytes once
    original_bytes = await file.read()
    
    print("✨ Removing background and normalizing...")
    # Decode to OpenCV format once for processing
    try:
        # GLOBAL PRE-PROCESSING: Remove Background & Make White
        # This ensures EVERY model sees the clean, isolated leaf.
        clean_leaf_cv2 = remove_background_add_white(original_bytes)
        # original_cv2 = read_image_from_bytes(original_bytes)
    except Exception as e:
        print(f"Background removal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")
    
    # Parallel Request Processing
    async with httpx.AsyncClient() as client:
        tasks = []
        for name, info in SERVICES.items():
            # We pass 'clean_leaf_cv2' instead of raw bytes decoding
            tasks.append(query_microservice(client, name, info, clean_leaf_cv2))
        
        # Wait for all to finish
        microservice_responses = await asyncio.gather(*tasks)
    
    # DECISION LOGIC
    # ---------------------------------------------------------
    if use_weather and lat is not None and lon is not None:
        print("Engaging Weather Engine...")
        
        # A. Get Temp (Real or Test Value)
        avg_temp = get_average_temperature(lat, lon)
        
        # B. Refine Prediction
        final_result = refine_prediction_by_weather(microservice_responses, avg_temp)
    else:
        print("Standard Majority Voting (No Weather)")
        # Import the old function logic or define it here
        final_result = calculate_voting_result(microservice_responses)
    # ---------------------------------------------------------
    # 4. 🟢 Save to Supabase (Async-ish)
    if user_id:
        # We perform this *after* getting the result so the user doesn't wait too long,
        # but in synchronous Python, it still blocks slightly. 
        # For production, use BackgroundTasks. For now, direct call is fine.
        save_prediction_to_db(user_id, original_bytes, final_result)
   
    return final_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=30)