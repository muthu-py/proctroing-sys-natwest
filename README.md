# Online Proctoring - Face Detection System

This system captures webcam input from a web browser and processes it using Python for face presence detection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:8000`

## How it works

1. **Frontend (index.html)**: Captures webcam video and sends frames to the backend via WebSocket
2. **Backend (app.py)**: Receives frames, processes them using the FacePresence class, and sends back detection results
3. **Face Detection (Facepresence.py)**: Uses MediaPipe to detect faces and flag violations

## Features

- Real-time webcam streaming
- Face presence detection
- Violation flagging (0 faces or multiple faces)
- WebSocket communication for low latency
- Visual status updates in the browser

## API Endpoints

- `GET /` - Serves the main HTML page
- `WebSocket /ws` - Real-time communication for frame processing
