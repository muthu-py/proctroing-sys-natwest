from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from face.deepverify.Facepresence import DeepFacePresence
from face.deepverify.host import DeepHostEmbedding

app = FastAPI()

# Serve static files (HTML, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DeepFace presence detector
face_detector = DeepFacePresence(model_name="Facenet", face_match_threshold=0.6)

# Initialize DeepFace host embedding processor
host_embedder = DeepHostEmbedding(model_name="Facenet")

# Global variable to track if photos have been uploaded
photos_uploaded = False
uploaded_photos_count = 0
average_embedding = None

@app.get("/", response_class=HTMLResponse)
async def index():
    # If photos haven't been uploaded, redirect to capture page
    if not photos_uploaded:
        return RedirectResponse(url="/capture", status_code=302)
    
    # If photos are uploaded, show the face detection page
    with open("static/index.html") as f:
        return f.read()

@app.get("/capture", response_class=HTMLResponse)
async def capture():
    """Show the photo capture page"""
    with open("static/capture.html") as f:
        return f.read()

@app.post("/upload")
async def upload_photos(photos: list[UploadFile] = File(...), ts: str = Form(...), device: str = Form(...)):
    """Handle photo uploads and process through DeepFace embedding module"""
    global photos_uploaded, uploaded_photos_count, average_embedding
    
    if len(photos) < 4:
        raise HTTPException(status_code=400, detail="At least 4 photos are required")
    
    try:
        # Read all photo data
        photos_data = []
        for photo in photos:
            content = await photo.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {photo.filename}")
            photos_data.append(content)
        
        # Process photos through DeepFace host embedding module
        print(f"Processing {len(photos_data)} photos through DeepFace host embedding module...")
        success = host_embedder.process_photos(photos_data)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to generate DeepFace embeddings from photos")
        
        # Get average embedding
        average_embedding = host_embedder.get_average_embedding()
        embedding_count = host_embedder.get_embedding_count()
        
        # Set reference embedding for face matching
        if average_embedding is not None:
            face_detector.set_reference_embedding(average_embedding)
        
        # Update global state
        photos_uploaded = True
        uploaded_photos_count = len(photos)
        
        print(f"Successfully processed {embedding_count} photos and generated DeepFace average embedding")
        print(f"Average embedding length: {len(average_embedding) if average_embedding else 0}")
        print("DeepFace reference embedding set for face matching")
        
        return {
            "success": True,
            "message": f"Successfully processed {embedding_count} photos and generated DeepFace embeddings",
            "photo_count": len(photos),
            "embedding_count": embedding_count,
            "embedding_length": len(average_embedding) if average_embedding else 0,
            "model_info": host_embedder.get_model_info(),
            "redirect_url": "/"
        }
        
    except Exception as e:
        print(f"DeepFace upload processing error: {e}")
        raise HTTPException(status_code=500, detail=f"DeepFace upload processing failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Check if photos have been uploaded and processed"""
    return {
        "photos_uploaded": photos_uploaded,
        "photo_count": uploaded_photos_count,
        "embedding_available": average_embedding is not None,
        "embedding_length": len(average_embedding) if average_embedding else 0,
        "model_info": host_embedder.get_model_info() if photos_uploaded else None
    }

@app.post("/reset")
async def reset_photos():
    """Reset uploaded photos and redirect to capture"""
    global photos_uploaded, uploaded_photos_count, average_embedding
    
    # Clear uploaded photos state
    photos_uploaded = False
    uploaded_photos_count = 0
    average_embedding = None
    
    # Clear embeddings from the host embedder
    host_embedder.clear_embeddings()
    
    # Clear reference embedding from face detector
    face_detector.set_reference_embedding(None)
    
    return {
        "success": True,
        "message": "DeepFace photos and embeddings reset successfully",
        "redirect_url": "/capture"
    }

@app.get("/embedding")
async def get_embedding():
    """Get the average embedding if available"""
    if not photos_uploaded or average_embedding is None:
        raise HTTPException(status_code=404, detail="No DeepFace embedding available. Please upload photos first.")
    
    return {
        "embedding": average_embedding,
        "embedding_length": len(average_embedding),
        "photo_count": uploaded_photos_count,
        "model_info": host_embedder.get_model_info()
    }

@app.get("/face-match-info")
async def get_face_match_info():
    """Get face matching information for debugging"""
    if not photos_uploaded:
        raise HTTPException(status_code=404, detail="No photos uploaded yet.")
    
    return {
        "photos_uploaded": photos_uploaded,
        "reference_embedding_available": average_embedding is not None,
        "face_match_info": face_detector.get_face_match_info(),
        "model_performance": face_detector.get_model_performance_info()
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    return {
        "current_model": host_embedder.model_name,
        "available_models": host_embedder.face_embedder.get_available_models(),
        "available_backends": host_embedder.face_embedder.get_available_backends(),
        "model_performance": face_detector.get_model_performance_info()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("DeepFace WebSocket connection established")
    
    try:
        while True:
            # Receive base64 encoded frame from frontend
            data = await websocket.receive_text()
            
            # Process frame using DeepFace Presence module
            response = face_detector.process_base64_frame(data)
            
            # Send response back to frontend
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("DeepFace WebSocket connection closed")
    except Exception as e:
        print(f"DeepFace WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    print("Starting DeepFace Proctoring System...")
    print("Model: Facenet")
    print("Threshold: 0.6")
    print("Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
