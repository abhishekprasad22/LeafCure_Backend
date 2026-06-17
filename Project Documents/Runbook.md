# Runbook

Use this runbook when starting, testing, or troubleshooting the backend locally.

## Start Order

1. Activate the virtual environment.
2. Start all five model services.
3. Start the main API service.
4. Test `POST /analyze_leaf`.

## Start Model Services

Run each command from the project root in a separate terminal:

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

## Start Main API

```powershell
python main.py
```

## Health Checks

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Check that every model service has started successfully. Each service should print that its model loaded successfully.

## Common Problems

### Model file not found

Confirm the file exists inside `models/` and the command is being run from the project root.

### Port already in use

Stop the process already using that port, or update both the model service command and the matching service URL in `main.py`.

### Supabase credentials not found

Create `.env` and add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. Predictions can still run without these credentials, but history saving will be disabled.

### Weather API failed

The backend falls back to `25.0 C`. Confirm that the machine has internet access and that latitude and longitude values are valid.

### Slow first request

TensorFlow model loading and `rembg` processing can make startup or first prediction slower. Keep services running during testing.

