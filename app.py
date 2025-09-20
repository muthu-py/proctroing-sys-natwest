from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import base64
import json
from face.Facepresence import FacePresence

app = FastAPI()

# Serve static files (HTML, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize face presence detector
face_detector = FacePresence()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            # Receive base64 encoded frame from frontend
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 image
            image_data = base64.b64decode(frame_data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame with face detection
            face_count, flag = face_detector.process_frame(frame)
            
            # Send response back to frontend
            response = {
                'face_count': face_count,
                'flag': flag,
                'timestamp': frame_data.get('timestamp')
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
