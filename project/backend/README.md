# HawkEye Backend

This backend powers the HawkEye frontend by processing thermal drone video feeds using YOLO and SAHI, then streaming annotated frame payloads over WebSockets.

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the backend:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API

- `GET /health` - detector status and configuration details
- `WS /stream` - live JSON stream of frame payloads

## Configuration

Change `backend/app/config.py` values or create a `.env` file in the `backend` folder.
