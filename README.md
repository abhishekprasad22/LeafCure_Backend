# LeafCure Backend

LeafCure Backend is a FastAPI based service for tea leaf disease detection. It accepts a leaf image, removes the image background, sends transformed image variants to five TensorFlow model microservices, combines their predictions with majority voting, and can optionally refine the result using recent weather data.

## Features

- FastAPI orchestrator endpoint for leaf analysis.
- Five TensorFlow model services:
  - Original image model
  - Grayscale model
  - Negative image model
  - False color model
  - Histogram equalized model
- Background removal before model inference.
- Majority voting across model predictions.
- Optional weather based disease refinement using Open-Meteo historical temperature data.
- Optional Supabase storage and prediction history saving.

## Project Structure

```text
LeafCure_Backend/
├── main.py
├── model_service.py
├── utils.py
├── weather_engine.py
├── requirements.txt
├── models/
│   ├── original_model.keras
│   ├── grayscale_model.keras
│   ├── negative_model.keras
│   ├── false_color_model.keras
│   └── histogram_model.keras
├── Project Documents/
│   ├── API_Documentation.md
│   ├── Architecture.md
│   ├── Environment.md
│   ├── Runbook.md
│   └── Set_up.md
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip
- TensorFlow compatible system environment
- The five `.keras` model files inside `models/`
- Supabase project credentials if prediction history should be saved

## Environment Variables

Create a local `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

Supabase is optional for running predictions. If credentials are missing, the API still runs, but prediction history will not be saved.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Model Services

Open five terminals from the project root and run one service in each terminal:

```powershell
python model_service.py --model_path models/original_model.keras --port 8001 --name Original
```

```powershell
python model_service.py --model_path models/grayscale_model.keras --port 8002 --name Grayscale
```

```powershell
python model_service.py --model_path models/negative_model.keras --port 8003 --name Negative
```

```powershell
python model_service.py --model_path models/false_color_model.keras --port 8004 --name FalseColor
```

```powershell
python model_service.py --model_path models/histogram_model.keras --port 8005 --name Histogram
```

## Run The Main API

In another terminal:

```powershell
python main.py
```

The main API runs at:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Main Endpoint

```http
POST /analyze_leaf
```

Multipart form fields:

| Field | Required | Description |
| --- | --- | --- |
| `file` | Yes | Leaf image file. |
| `use_weather` | No | Set to `true` to apply weather based refinement. |
| `lat` | Required when `use_weather=true` | Latitude for weather lookup. |
| `lon` | Required when `use_weather=true` | Longitude for weather lookup. |
| `user_id` | No | Supabase user id used for saving prediction history. |

## Documentation

- [Setup Guide](Project%20Documents/Set_up.md)
- [API Documentation](Project%20Documents/API_Documentation.md)
- [Architecture](Project%20Documents/Architecture.md)
- [Environment](Project%20Documents/Environment.md)
- [Runbook](Project%20Documents/Runbook.md)

