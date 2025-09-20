import cv2
import numpy as np
from typing import Optional, List, Tuple
import time
from face.deepverify.embb import DeepFaceEmbedding
from face.deepverify.simi import DeepSimilarityCalculator

class DeepFaceMatcher:
    def __init__(self, model_name: str = "Facenet", threshold: float = None):
        """
        Initialize DeepFace matcher
        
        Args:
            model_name: DeepFace model to use ('Facenet', 'VGG-Face', 'OpenFace', 'ArcFace', 'Dlib')
            threshold: Similarity threshold (uses model-specific if None)
        """
        self.model_name = model_name
        self.last_match_time = 0
        self.match_interval = 2  # 2 seconds
        
        # Initialize DeepFace embedder and similarity calculator
        self.face_embedder = DeepFaceEmbedding(model_name=model_name)
        self.similarity_calculator = DeepSimilarityCalculator()
        
        # Set threshold (use model-specific if not provided)
        if threshold is None:
            thresholds = self.similarity_calculator.get_similarity_thresholds()
            self.threshold = thresholds.get(model_name, 0.6)
        else:
            self.threshold = threshold
        
        # Store reference embedding
        self.reference_embedding = None
        
        print(f"DeepFace matcher initialized with model: {model_name}, threshold: {self.threshold}")

    def set_reference_embedding(self, embedding: List[float]):
        """Set the reference embedding from host.py"""
        self.reference_embedding = embedding
        print(f"DeepFace reference embedding set with length: {len(embedding)}")
        print(f"Reference embedding first 5 values: {embedding[:5]}")

    def should_check_face(self) -> bool:
        """Check if enough time has passed for the next face check"""
        current_time = time.time()
        return (current_time - self.last_match_time) >= self.match_interval

    def match_face(self, frame) -> Tuple[bool, float, str]:
        """
        Check if the face in the frame matches the reference embedding using DeepFace
        
        Args:
            frame: OpenCV frame containing a face
            
        Returns:
            Tuple of (match_result, similarity_score, status)
        """
        if self.reference_embedding is None:
            print("No DeepFace reference embedding set")
            return True, 1.0, "no_reference"
        
        if not self.should_check_face():
            return True, 1.0, "not_checked"
        
        # Get embedding from current frame using DeepFace embb.py
        current_embedding = self.face_embedder.get_embedding(frame)
        
        if current_embedding is None:
            print("Failed to generate DeepFace embedding from current frame")
            return True, 1.0, "embedding_failed"
        
        # Calculate similarity using DeepFace simi.py
        similarity = self.similarity_calculator.calculate_similarity(
            current_embedding, 
            self.reference_embedding
        )
        
        # Update last match time
        self.last_match_time = time.time()
        
        # Check if similarity exceeds threshold
        match_result = similarity >= self.threshold
        
        print(f"DeepFace matching - Similarity: {similarity:.4f}, Threshold: {self.threshold}, Match: {match_result}")
        
        return match_result, similarity, "checked"

    def test_face_matching(self, frame) -> dict:
        """Test face matching with detailed debugging information"""
        if self.reference_embedding is None:
            return {"error": "No DeepFace reference embedding set"}
        
        # Get embedding from current frame
        current_embedding = self.face_embedder.get_embedding(frame)
        
        if current_embedding is None:
            return {"error": "Failed to generate DeepFace embedding from current frame"}
        
        # Calculate similarity
        similarity = self.similarity_calculator.calculate_similarity(
            current_embedding, 
            self.reference_embedding
        )
        
        # Get detailed comparison
        comparison = self.similarity_calculator.compare_embeddings(
            current_embedding, 
            self.reference_embedding
        )
        
        # Check if similarity exceeds threshold
        match_result = similarity >= self.threshold
        
        return {
            "match_result": match_result,
            "similarity": similarity,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "current_embedding_length": len(current_embedding),
            "reference_embedding_length": len(self.reference_embedding),
            "detailed_comparison": comparison
        }

    def verify_faces_direct(self, frame1, frame2) -> dict:
        """
        Directly verify two frames using DeepFace's built-in verification
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Dictionary with verification result
        """
        try:
            result = self.face_embedder.verify_faces(frame1, frame2)
            return result
        except Exception as e:
            return {
                "verified": False,
                "distance": float('inf'),
                "threshold": 0.0,
                "similarity": 0.0,
                "error": str(e)
            }

    def update_threshold(self, new_threshold: float):
        """Update the similarity threshold"""
        self.threshold = max(0.0, min(1.0, new_threshold))
        print(f"Updated DeepFace similarity threshold to: {self.threshold}")

    def get_match_info(self) -> dict:
        """Get current matching information"""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "reference_embedding_length": len(self.reference_embedding) if self.reference_embedding else None,
            "last_match_time": self.last_match_time,
            "next_check_in": max(0, self.match_interval - (time.time() - self.last_match_time)),
            "available_models": self.face_embedder.get_available_models(),
            "available_backends": self.face_embedder.get_available_backends()
        }

    def get_model_performance_info(self) -> dict:
        """Get information about model performance and recommendations"""
        return {
            "current_model": self.model_name,
            "current_threshold": self.threshold,
            "recommended_thresholds": self.similarity_calculator.get_similarity_thresholds(),
            "model_characteristics": {
                "Facenet": "Good balance of accuracy and speed, 128 dimensions",
                "VGG-Face": "High accuracy, slower, 4096 dimensions", 
                "OpenFace": "Fast, good for real-time, 128 dimensions",
                "ArcFace": "Very high accuracy, slower, 512 dimensions",
                "Dlib": "Fast, good for real-time, 128 dimensions"
            }
        }

    def release(self):
        """Release resources"""
        self.face_embedder.release()


# Example usage for testing
if __name__ == "__main__":
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam")
        exit()
    
    # Create DeepFace matcher
    face_matcher = DeepFaceMatcher(model_name="Facenet", threshold=0.6)
    
    # Simulate a reference embedding (in real usage, this comes from host.py)
    dummy_reference = [0.5] * 128  # 128 dimensions for Facenet
    face_matcher.set_reference_embedding(dummy_reference)
    
    print("DeepFace matcher initialized. Press 'q' to quit.")
    print(f"Model info: {face_matcher.get_model_performance_info()}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check for face match
        match, similarity, status = face_matcher.match_face(frame)
        
        if match:
            print(f"✅ DeepFace match detected! Similarity: {similarity:.4f}")
        else:
            print(f"❌ No DeepFace match. Similarity: {similarity:.4f}")
        
        # Display frame
        cv2.putText(frame, f"DeepFace Similarity: {similarity:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Match: {match}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if match else (0, 0, 255), 2)
        cv2.putText(frame, f"Model: {face_matcher.model_name}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("DeepFace Matcher", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    face_matcher.release()
    cap.release()
    cv2.destroyAllWindows()
