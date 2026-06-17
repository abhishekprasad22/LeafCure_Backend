# API Documentation

This backend exposes a FastAPI application from `main.py`. The main service acts as an orchestrator: it receives the uploaded leaf image, prepares transformed image versions, calls each model microservice, and returns the final voted prediction.

## Base URL

```text
http://127.0.0.1:8000
```

## Interactive Docs

FastAPI automatically provides Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Analyze Leaf

```http
POST /analyze_leaf
```

### Request Type

`multipart/form-data`

### Form Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | File | Yes | Tea leaf image to analyze. |
| `use_weather` | Boolean | No | Enables weather based refinement when set to `true`. Default is `false`. |
| `lat` | Float | Required if `use_weather=true` | Latitude used for Open-Meteo weather lookup. |
| `lon` | Float | Required if `use_weather=true` | Longitude used for Open-Meteo weather lookup. |
| `user_id` | String | No | Supabase user id. If provided, prediction history is saved. |

### Example Without Weather

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze_leaf" `
  -F "file=@sample_leaf.jpg" `
  -F "use_weather=false"
```

### Example With Weather

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze_leaf" `
  -F "file=@sample_leaf.jpg" `
  -F "use_weather=true" `
  -F "lat=26.1445" `
  -F "lon=91.7362"
```

### Example With History Saving

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze_leaf" `
  -F "file=@sample_leaf.jpg" `
  -F "use_weather=true" `
  -F "lat=26.1445" `
  -F "lon=91.7362" `
  -F "user_id=your-user-id"
```

## Successful Response

```json
{
  "final_prediction": "healthy",
  "final_confidence": 96.42,
  "vote_count": 3,
  "total_models": 5,
  "details": [
    {
      "model_name": "Original",
      "prediction": "healthy",
      "confidence": 97.1,
      "used_transform": "original"
    }
  ]
}
```

When weather refinement is enabled, the response also includes:

```json
{
  "weather_data": {
    "avg_temp_30d": 25.8,
    "used_weather_filter": true
  }
}
```

## Error Response

If no model service returns a valid prediction:

```json
{
  "error": "No successful predictions from microservices"
}
```

If background removal fails, the API returns HTTP `500` with a failure message.

## Model Service Endpoint

Each model service created by `model_service.py` exposes:

```http
POST /predict
```

This endpoint receives one transformed image file and returns:

```json
{
  "model_name": "Original",
  "prediction": "healthy",
  "confidence": 96.42
}
```

The orchestrator calls these services internally. Frontend clients should call `/analyze_leaf`, not the individual `/predict` endpoints.

