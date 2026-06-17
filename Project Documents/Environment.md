# Environment

This file documents local configuration, credentials, and external services used by the backend.

## Local `.env`

Create `.env` in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

Do not commit `.env`. It contains private credentials and is ignored by Git.

## Supabase

Supabase is used only when saving prediction history. The app can still analyze images without Supabase credentials.

Expected Supabase resources:

| Resource | Name | Purpose |
| --- | --- | --- |
| Storage bucket | `leaf_images` | Stores uploaded leaf image files. |
| Database table | `predictions` | Stores prediction history and analysis result JSON. |

The `predictions` insert in `main.py` expects these fields:

| Field | Source |
| --- | --- |
| `user_id` | Request form field |
| `image_path` | Generated storage path |
| `prediction` | Final disease prediction |
| `confidence` | Final confidence score |
| `analysis_data` | Full result object |

## Weather API

Weather data is fetched from Open-Meteo Archive API:

```text
https://archive-api.open-meteo.com/v1/archive
```

No API key is required. The backend requests the previous 30 days of daily mean temperature and calculates an average.

If the weather request fails or returns no temperature data, the backend falls back to `25.0 C`.

## Model Files

The application expects these runtime model files:

```text
models/original_model.keras
models/grayscale_model.keras
models/negative_model.keras
models/false_color_model.keras
models/histogram_model.keras
```

Keep model filenames aligned with the startup commands in `Set_up.md` and the service map in `main.py`.

