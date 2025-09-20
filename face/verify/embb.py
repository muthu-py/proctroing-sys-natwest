import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List

class FaceEmbedding:
    def __init__(self):
        """Initialize face mesh for embedding generation"""
        # Initialize face mesh for detailed face analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_face_features(self, image):
        """Extract face features using MediaPipe face mesh"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key facial features (normalized coordinates)
            features = []
            for landmark in face_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            return features
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None

    def get_embedding(self, frame) -> Optional[List[float]]:
        """
        Get embedding for a single frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of float values representing the face embedding (1434 dimensions)
            Returns None if no face detected or error occurred
        """
        try:
            # Extract face features
            features = self.extract_face_features(frame)
            
            if features is None:
                print("No face detected in frame")
                return None
            
            print(f"Generated embedding with {len(features)} dimensions")
            return features
                    
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def release(self):
        """Release resources"""
        self.face_mesh.close()


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
    
    # Test embedding generation
    face_embedder = FaceEmbedding()
    embedding = face_embedder.get_embedding(frame)
    
    if embedding:
        print("Test embedding generated successfully")
        print(f"Embedding length: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
    else:
        print("Failed to generate test embedding")
    
    face_embedder.release()
    cap.release()
