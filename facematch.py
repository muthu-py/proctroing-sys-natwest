import cv2
import numpy as np
from typing import Optional, List, Tuple
import time
from face.verify.embb import FaceEmbedding
from face.verify.simi import SimilarityCalculator

class FaceMatcher:
    def __init__(self, threshold: float = 0.85):
        """
        Initialize face matcher
        
        Args:
            threshold: Similarity threshold for face matching (0.0 to 1.0)
        """
        self.threshold = threshold
        self.last_match_time = 0
        self.match_interval = 2  # 2 seconds
        
        # Initialize face embedder and similarity calculator
        self.face_embedder = FaceEmbedding()
        self.similarity_calculator = SimilarityCalculator()
        
        # Store reference embedding
        self.reference_embedding = None

    def set_reference_embedding(self, embedding: List[float]):
        """Set the reference embedding from host.py"""
        self.reference_embedding = embedding
        print(f"Reference embedding set with length: {len(embedding)}")
        print(f"Reference embedding first 5 values: {embedding[:5]}")

    def should_check_face(self) -> bool:
        """Check if enough time has passed for the next face check"""
        current_time = time.time()
        return (current_time - self.last_match_time) >= self.match_interval

    def match_face(self, frame) -> Tuple[bool, float, str]:
        """
        Check if the face in the frame matches the reference embedding
        
        Args:
            frame: OpenCV frame containing a face
            
        Returns:
            Tuple of (match_result, similarity_score, status)
        """
        if self.reference_embedding is None:
            print("No reference embedding set")
            return True, 1.0, "no_reference"
        
        if not self.should_check_face():
            return True, 1.0, "not_checked"
        
        # Get embedding from current frame using embb.py
        current_embedding = self.face_embedder.get_embedding(frame)
        
        if current_embedding is None:
            print("Failed to generate embedding from current frame")
            return True, 1.0, "embedding_failed"
        
        # Calculate similarity using simi.py
        similarity = self.similarity_calculator.calculate_similarity(
            current_embedding, 
            self.reference_embedding
        )
        
        # Update last match time
        self.last_match_time = time.time()
        
        # Check if similarity exceeds threshold
        match_result = similarity >= self.threshold
        
        print(f"Face matching - Similarity: {similarity:.4f}, Threshold: {self.threshold}, Match: {match_result}")
        
        return match_result, similarity, "checked"

    def test_face_matching(self, frame) -> dict:
        """Test face matching with detailed debugging information"""
        if self.reference_embedding is None:
            return {"error": "No reference embedding set"}
        
        # Get embedding from current frame
        current_embedding = self.face_embedder.get_embedding(frame)
        
        if current_embedding is None:
            return {"error": "Failed to generate embedding from current frame"}
        
        # Calculate similarity
        similarity = self.similarity_calculator.calculate_similarity(
            current_embedding, 
            self.reference_embedding
        )
        
        # Check if similarity exceeds threshold
        match_result = similarity >= self.threshold
        
        return {
            "match_result": match_result,
            "similarity": similarity,
            "threshold": self.threshold,
            "current_embedding_length": len(current_embedding),
            "reference_embedding_length": len(self.reference_embedding)
        }

    def update_threshold(self, new_threshold: float):
        """Update the similarity threshold"""
        self.threshold = max(0.0, min(1.0, new_threshold))
        print(f"Updated similarity threshold to: {self.threshold}")

    def get_match_info(self) -> dict:
        """Get current matching information"""
        return {
            "threshold": self.threshold,
            "reference_embedding_length": len(self.reference_embedding) if self.reference_embedding else None,
            "last_match_time": self.last_match_time,
            "next_check_in": max(0, self.match_interval - (time.time() - self.last_match_time))
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
    
    # Create face matcher
    face_matcher = FaceMatcher(threshold=0.85)
    
    # Simulate a reference embedding (in real usage, this comes from host.py)
    dummy_reference = [0.5] * 1434  # 1434 dimensions
    face_matcher.set_reference_embedding(dummy_reference)
    
    print("Face matcher initialized. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check for face match
        match, similarity, status = face_matcher.match_face(frame)
        
        if match:
            print(f"✅ Face match detected! Similarity: {similarity:.4f}")
        else:
            print(f"❌ No face match. Similarity: {similarity:.4f}")
        
        # Display frame
        cv2.putText(frame, f"Similarity: {similarity:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Match: {match}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if match else (0, 0, 255), 2)
        
        cv2.imshow("Face Matcher", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    face_matcher.release()
    cap.release()
    cv2.destroyAllWindows()
