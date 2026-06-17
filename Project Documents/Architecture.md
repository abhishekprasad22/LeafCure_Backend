# Architecture

LeafCure Backend uses an orchestrator and microservice pattern. The main API receives one image from the frontend, prepares the image, sends variants to model services, and combines the model outputs into one final result.

## Components

| Component | File | Responsibility |
| --- | --- | --- |
| API orchestrator | `main.py` | Receives uploads, removes background, calls model services, applies voting, saves history. |
| Model service runtime | `model_service.py` | Loads one `.keras` model and exposes `/predict`. |
| Image utilities | `utils.py` | Decodes images, removes background, applies image transformations. |
| Weather engine | `weather_engine.py` | Fetches historical weather and refines predictions based on temperature compatibility. |
| Model files | `models/` | Stores trained TensorFlow/Keras disease classification models. |
| Supabase | External service | Stores uploaded image paths and prediction records when `user_id` is provided. |
| Open-Meteo | External API | Provides recent historical temperature data for weather based filtering. |

## Runtime Flow

1. The frontend sends a leaf image to `POST /analyze_leaf`.
2. `main.py` reads the uploaded image bytes.
3. `remove_background_add_white` removes the image background and replaces it with white.
4. `main.py` creates five inference paths:
   - original
   - grayscale
   - negative
   - false color
   - histogram equalized
5. Each image version is sent to its matching model service.
6. Each model returns a disease label and confidence score.
7. `calculate_voting_result` selects the majority prediction.
8. If weather mode is enabled, `refine_prediction_by_weather` checks the winning disease against the last 30 days of average temperature.
9. If `user_id` is supplied and Supabase is configured, the image and result are saved.
10. The API returns the final result to the frontend.

## Ports

| Port | Service |
| --- | --- |
| `8000` | Main FastAPI orchestrator |
| `8001` | Original image model |
| `8002` | Grayscale model |
| `8003` | Negative image model |
| `8004` | False color model |
| `8005` | Histogram equalized model |

## Disease Classes

The model services currently use these class names:

```text
algal_spot
brown_blight
gray_blight
healthy
helopeltis
red_spot
```

## Weather Refinement

The weather engine compares predicted disease classes against configured temperature ranges:

| Disease | Temperature Range |
| --- | --- |
| `healthy` | 13 C to 30 C |
| `algal_spot` | 20 C to 35 C |
| `brown_blight` | 15 C to 30 C |
| `gray_blight` | 25 C to 38 C |
| `helopeltis` | 15 C to 30 C |
| `red_spot` | 15 C to 32 C |

Weather filtering only changes the final result when the majority prediction is incompatible with the recent average temperature and a runner-up prediction is compatible.

