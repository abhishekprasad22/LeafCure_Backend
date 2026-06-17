# Setup Guide

This guide explains how to install and run the LeafCure backend locally.

## 1. Create Virtual Environment

Run these commands from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For Git Bash or Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

These values are required only for saving prediction history. The prediction endpoint can run without Supabase.

## 4. Verify Model Files

Confirm that these files exist:

```text
models/original_model.keras
models/grayscale_model.keras
models/negative_model.keras
models/false_color_model.keras
models/histogram_model.keras
```

## 5. Start Model Services

Open five terminals from the project root and run one command in each terminal.

Terminal 1:

```powershell
python model_service.py --model_path models/original_model.keras --port 8001 --name Original
```

Terminal 2:

```powershell
python model_service.py --model_path models/grayscale_model.keras --port 8002 --name Grayscale
```

Terminal 3:

```powershell
python model_service.py --model_path models/negative_model.keras --port 8003 --name Negative
```

Terminal 4:

```powershell
python model_service.py --model_path models/false_color_model.keras --port 8004 --name FalseColor
```

Terminal 5:

```powershell
python model_service.py --model_path models/histogram_model.keras --port 8005 --name Histogram
```

## 6. Start Main API

Open another terminal from the project root:

```powershell
python main.py
```

The main API starts at:

```text
http://127.0.0.1:8000
```

## 7. Test The API

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/analyze_leaf` endpoint to upload a leaf image.

## How The Backend Works

1. The user uploads one leaf image to `main.py` on port `8000`.
2. The backend removes the background and normalizes the image.
3. The backend creates transformed versions of the image using `utils.py`.
4. The backend sends the image versions to model services on ports `8001` to `8005`.
5. Each model returns a disease prediction and confidence score.
6. The backend applies majority voting.
7. If weather mode is enabled, the backend checks the result against recent average temperature.
8. The final result is returned to the client.
