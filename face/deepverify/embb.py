import cv2
import numpy as np
from deepface import DeepFace
from typing import Optional, List
import os

class DeepFaceEmbedding:
    def __init__(self, model_name: str = "Facenet"):
        """
        Initialize DeepFace embedding generator
        
        Args:
            model_name: DeepFace model to use ('Facenet', 'VGG-Face', 'OpenFace', 'ArcFace', 'Dlib')
        """
        self.model_name = model_name
        self.backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        
        # Pre-load the model to avoid loading time during inference
        print(f"Initializing DeepFace with model: {model_name}")
        try:
            # Test with a dummy image to pre-load the model
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(dummy_img, model_name=model_name, enforce_detection=False)
            print(f"DeepFace model {model_name} loaded successfully")
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")

    def get_embedding(self, frame) -> Optional[List[float]]:
        """
        Get face embedding using DeepFace
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of float values representing the face embedding
            Returns None if no face detected or error occurred
        """
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=rgb_frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend=self.backends[0]  # Use opencv as default
            )
            
            if embedding and len(embedding) > 0:
                # Extract the embedding vector
                embedding_vector = embedding[0]["embedding"]
                print(f"Generated DeepFace embedding with {len(embedding_vector)} dimensions using {self.model_name}")
                return embedding_vector
            else:
                print("No face detected in frame")
                return None
                
        except Exception as e:
            print(f"Error generating DeepFace embedding: {e}")
            # Try with different backends if the first one fails
            for backend in self.backends[1:]:
                try:
                    print(f"Retrying with backend: {backend}")
                    embedding = DeepFace.represent(
                        img_path=rgb_frame,
                        model_name=self.model_name,
                        enforce_detection=True,
                        detector_backend=backend
                    )
                    
                    if embedding and len(embedding) > 0:
                        embedding_vector = embedding[0]["embedding"]
                        print(f"Generated DeepFace embedding with {len(embedding_vector)} dimensions using {backend}")
                        return embedding_vector
                        
                except Exception as backend_error:
                    print(f"Backend {backend} failed: {backend_error}")
                    continue
            
            return None

    def verify_faces(self, img1, img2) -> dict:
        """
        Verify if two images contain the same person using DeepFace
        
        Args:
            img1: First image (OpenCV frame)
            img2: Second image (OpenCV frame)
            
        Returns:
            Dictionary with verification result and similarity score
        """
        try:
            # Convert BGR to RGB
            rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Verify faces
            result = DeepFace.verify(
                img1_path=rgb1,
                img2_path=rgb2,
                model_name=self.model_name,
                detector_backend=self.backends[0],
                distance_metric="cosine"
            )
            
            return {
                "verified": result["verified"],
                "distance": result["distance"],
                "threshold": result["threshold"],
                "similarity": 1 - result["distance"]  # Convert distance to similarity
            }
            
        except Exception as e:
            print(f"Error verifying faces: {e}")
            return {
                "verified": False,
                "distance": float('inf'),
                "threshold": 0.0,
                "similarity": 0.0,
                "error": str(e)
            }

    def find_faces(self, frame) -> List[dict]:
        """
        Find all faces in the frame and return their locations
        
        Args:
            frame: OpenCV frame
            
        Returns:
            List of dictionaries with face information
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract faces
            faces = DeepFace.extract_faces(
                img=rgb_frame,
                detector_backend=self.backends[0],
                enforce_detection=False
            )
            
            return faces
            
        except Exception as e:
            print(f"Error finding faces: {e}")
            return []

    def get_available_models(self) -> List[str]:
        """Get list of available DeepFace models"""
        return ["Facenet", "VGG-Face", "OpenFace", "ArcFace", "Dlib"]

    def get_available_backends(self) -> List[str]:
        """Get list of available detector backends"""
        return self.backends

    def release(self):
        """Release resources (DeepFace doesn't need explicit cleanup)"""
        print("DeepFace resources released")


# Example usage for testing
if __name__ == "__main__":
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam")
        exit()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        cap.release()
        exit()
    
    # Test DeepFace embedding generation
    deep_embedder = DeepFaceEmbedding(model_name="Facenet")
    embedding = deep_embedder.get_embedding(frame)
    
    if embedding:
        print("DeepFace embedding generated successfully")
        print(f"Embedding length: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Available models: {deep_embedder.get_available_models()}")
    else:
        print("Failed to generate DeepFace embedding")
    
    deep_embedder.release()
    cap.release()
