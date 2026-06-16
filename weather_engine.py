import requests
import datetime
from collections import Counter

# --- 1. CONFIGURATION TABLE ---
# The ranges you provided
DISEASE_TEMP_RANGES = {
    "healthy":      (13, 30),
    "algal_spot":   (20, 35),
    "brown_blight": (15, 30),
    "gray_blight":  (25, 38),
    "helopeltis":   (15, 30),
    "red_spot":     (15, 32), # "Red Spot" mapped to code style "red_spot"
}

# --- 2. TESTING CONFIGURATION ---
# SET THIS TO A NUMBER (e.g., 36.0) TO FORCE A TEST TEMP
# SET THIS TO None TO USE REAL GPS WEATHER DATA
TEST_OVERRIDE_TEMP = None

def get_average_temperature(lat, lon):
    """
    Fetches historical weather for the last 30 days from Open-Meteo 
    and returns the average temperature.
    """
    # TESTING HOOK: If we set a test value, return it immediately
    if TEST_OVERRIDE_TEMP is not None:
        print(f"🧪 [TEST MODE] Using Hardcoded Temp: {TEST_OVERRIDE_TEMP}°C")
        return TEST_OVERRIDE_TEMP

    print(f"Fetching weather for Lat: {lat}, Lon: {lon}...")
    
    try:
        # Calculate dates
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)
        
        # Open-Meteo Archive API (Free, no key needed)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start_date),
            "end_date": str(today),
            "daily": "temperature_2m_mean", # Daily average temp
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Calculate Average
        temps = data.get("daily", {}).get("temperature_2m_mean", [])
        # Filter out None values just in case
        valid_temps = [t for t in temps if t is not None]
        
        if not valid_temps:
            print("⚠️ No temp data found, returning default safe value (25°C)")
            return 25.0
            
        avg_temp = sum(valid_temps) / len(valid_temps)
        print(f"Calculated Avg Temp (30 days): {avg_temp:.2f}°C")
        return avg_temp

    except Exception as e:
        print(f"Weather API Failed: {e}")
        # Return a 'safe' temp that fits most ranges if API fails
        return 25.0 

def is_compatible(disease_name, temp):
    """Checks if the temp is within the disease's range."""
    # Normalize name (handle formatting differences)
    norm_name = disease_name.lower().replace(" ", "_")
    
    # Try to find partial matches if exact key missing
    # e.g., "Algal Leaf Spot" -> matches "algal_spot" key
    range_limit = None
    for key, val in DISEASE_TEMP_RANGES.items():
        if key in norm_name or norm_name in key:
            range_limit = val
            break
            
    if not range_limit:
        print(f"Unknown disease '{disease_name}', assuming compatible.")
        return True # Default to True if we don't know the disease
        
    min_t, max_t = range_limit
    return min_t <= temp <= max_t

def refine_prediction_by_weather(predictions, avg_temp):
    """
    1. Sorts predictions by vote count.
    2. Checks the top winner against temp.
    3. If invalid, moves to the next runner-up.
    """
    valid_predictions = [p for p in predictions if "error" not in p]
    if not valid_predictions:
        return None

    # 1. Count votes
    votes = Counter(p['prediction'] for p in valid_predictions)
    
    # 2. Sort candidates by Votes (Desc) then by Confidence (Desc)
    # We create a list of candidates: ['Algal Spot', 'Healthy', ...] ordered by rank
    ranked_candidates = votes.most_common() # Returns [('Healthy', 3), ('Algal Spot', 2)]
    
    print(f"Initial Voting Standings: {ranked_candidates}")
    print(f"Checking against Temp: {avg_temp}°C")

    # 3. Iterate through candidates to find the first 'Weather Compatible' one
    final_winner = None
    
    for disease, count in ranked_candidates:
        if is_compatible(disease, avg_temp):
            final_winner = disease
            print(f"ACCEPTED '{disease}' (Compatible with {avg_temp}°C)")
            break
        else:
            print(f"REJECTED '{disease}' (Incompatible with {avg_temp}°C)")
            
    # Fallback: If ALL are incompatible (rare), keep the original top voter
    if final_winner is None:
        print("No compatible disease found. Reverting to majority vote.")
        final_winner = ranked_candidates[0][0]

    # 4. Recalculate stats for the NEW winner
    winning_confidences = [p['confidence'] for p in valid_predictions if p['prediction'] == final_winner]
    avg_conf = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0

    return {
        "final_prediction": final_winner,
        "final_confidence": round(avg_conf, 2),
        "vote_count": votes[final_winner],
        "total_models": len(valid_predictions),
        "details": valid_predictions,
        "weather_data": {
            "avg_temp_30d": round(avg_temp, 1),
            "used_weather_filter": True
        }
    }


def calculate_voting_result(predictions):
    """
    Performs Standard Maximum Voting (Majority Rule) without Weather Logic.
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
    winning_confidences = [p['confidence'] for p in valid_predictions if p['prediction'] == top_prediction]
    avg_confidence = sum(winning_confidences) / len(winning_confidences)

    return {
        "final_prediction": top_prediction,
        "final_confidence": round(avg_confidence, 2),
        "vote_count": top_vote_count,
        "total_models": len(valid_predictions),
        "details": valid_predictions 
    }