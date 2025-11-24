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