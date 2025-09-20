from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from face.Facepresence_simple import SimpleFacePresence

app = FastAPI()

# Serve static files (HTML, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize simple face presence detector
face_detector = SimpleFacePresence()

# Global variable to track if photo has been uploaded
photo_uploaded = False
uploaded_photo_count = 0

@app.get("/", response_class=HTMLResponse)
async def index():
    # If photo hasn't been uploaded, redirect to capture page
    if not photo_uploaded:
        return RedirectResponse(url="/capture", status_code=302)
    
    # If photo is uploaded, show the face detection page
    with open("static/index.html") as f:
        return f.read()

@app.get("/capture", response_class=HTMLResponse)
async def capture():
    """Show the simple photo capture page"""
    with open("static/simple_capture.html") as f:
        return f.read()

@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...), ts: str = Form(...), device: str = Form(...)):
    """Handle single photo upload and set as reference"""
    global photo_uploaded, uploaded_photo_count
    
    try:
        # Read photo data
        content = await photo.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Empty file: {photo.filename}")
        
        # Set reference photo for face verification
        face_detector.set_reference_photo(content)
        
        # Update global state
        photo_uploaded = True
        uploaded_photo_count = 1
        
        print(f"Successfully uploaded reference photo: {photo.filename}")
        print("Reference photo set for face verification")
        
        return {
            "success": True,
            "message": f"Successfully uploaded reference photo: {photo.filename}",
            "photo_count": 1,
            "redirect_url": "/"
        }
        
    except Exception as e:
        print(f"Photo upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Photo upload failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Check if photo has been uploaded and processed"""
    return {
        "photo_uploaded": photo_uploaded,
        "photo_count": uploaded_photo_count,
        "verification_info": face_detector.get_verification_info()
    }

@app.post("/reset")
async def reset_photo():
    """Reset uploaded photo and redirect to capture"""
    global photo_uploaded, uploaded_photo_count
    
    # Clear uploaded photo state
    photo_uploaded = False
    uploaded_photo_count = 0
    
    # Clear reference photo from face detector
    face_detector.reference_photo = None
    
    return {
        "success": True,
        "message": "Photo reset successfully",
        "redirect_url": "/capture"
    }

@app.get("/verification-info")
async def get_verification_info():
    """Get face verification information for debugging"""
    if not photo_uploaded:
        raise HTTPException(status_code=404, detail="No photo uploaded yet.")
    
    return {
        "photo_uploaded": photo_uploaded,
        "reference_photo_available": face_detector.reference_photo is not None,
        "verification_info": face_detector.get_verification_info()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Simple Face Verification WebSocket connection established")
    
    try:
        while True:
            # Receive base64 encoded frame from frontend
            data = await websocket.receive_text()
            
            # Process frame using Simple Face Presence module
            response = face_detector.process_base64_frame(data)
            
            # Send response back to frontend
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("Simple Face Verification WebSocket connection closed")
    except Exception as e:
        print(f"Simple Face Verification WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    print("Starting Simple Face Verification System...")
    print("Using DeepFace direct verification")
    print("Model: Facenet")
    print("Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
