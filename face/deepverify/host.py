import numpy as np
from typing import List, Optional
from face.deepverify.embb import DeepFaceEmbedding
import cv2

class DeepHostEmbedding:
    def __init__(self, model_name: str = "Facenet"):
        """
        Initialize DeepFace host for managing multiple photo embeddings
        
        Args:
            model_name: DeepFace model to use ('Facenet', 'VGG-Face', 'OpenFace', 'ArcFace', 'Dlib')
        """
        self.face_embedder = DeepFaceEmbedding(model_name=model_name)
        self.embeddings = []
        self.average_embedding = None
        self.photo_count = 0
        self.model_name = model_name

    def process_photos(self, photos_data: List[bytes]) -> bool:
        """
        Process multiple photos and calculate average embedding using DeepFace
        
        Args:
            photos_data: List of photo data as bytes
            
        Returns:
            True if successful, False otherwise
        """
        self.embeddings = []
        self.photo_count = len(photos_data)
        
        print(f"Processing {self.photo_count} photos using DeepFace {self.model_name}...")
        
        for i, photo_data in enumerate(photos_data):
            print(f"Processing photo {i+1}/{self.photo_count}")
            
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(photo_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Failed to decode photo {i+1}")
                continue
            
            # Get embedding from DeepFace embb.py
            embedding = self.face_embedder.get_embedding(image)
            
            if embedding is not None:
                self.embeddings.append(embedding)
                print(f"Generated DeepFace embedding for photo {i+1} with {len(embedding)} dimensions")
            else:
                print(f"Failed to generate DeepFace embedding for photo {i+1}")
        
        if len(self.embeddings) > 0:
            self.calculate_average_embedding()
            return True
        else:
            print("No valid DeepFace embeddings generated")
            return False

    def calculate_average_embedding(self):
        """Calculate average embedding from all stored DeepFace embeddings"""
        if not self.embeddings:
            self.average_embedding = None
            return
        
        # Convert to numpy array for easier calculation
        embeddings_array = np.array(self.embeddings)
        
        # Calculate mean across all embeddings
        self.average_embedding = np.mean(embeddings_array, axis=0).tolist()
        
        print(f"Calculated average DeepFace embedding from {len(self.embeddings)} photos")
        print(f"Average embedding shape: {len(self.average_embedding)}")
        print(f"Model used: {self.model_name}")
        print(f"First 10 values: {self.average_embedding[:10]}")

    def get_average_embedding(self) -> Optional[List[float]]:
        """Get the calculated average embedding"""
        return self.average_embedding

    def get_embedding_count(self) -> int:
        """Get number of successful embeddings"""
        return len(self.embeddings)

    def get_photo_count(self) -> int:
        """Get total number of photos processed"""
        return self.photo_count

    def get_model_info(self) -> dict:
        """Get information about the model being used"""
        return {
            "model_name": self.model_name,
            "embedding_dimensions": len(self.average_embedding) if self.average_embedding else 0,
            "available_models": self.face_embedder.get_available_models(),
            "available_backends": self.face_embedder.get_available_backends()
        }

    def verify_photo(self, photo_data: bytes) -> dict:
        """
        Verify a single photo against the average embedding
        
        Args:
            photo_data: Photo data as bytes
            
        Returns:
            Dictionary with verification result
        """
        if self.average_embedding is None:
            return {"error": "No reference embedding available"}
        
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(photo_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode photo"}
        
        # Get embedding for the photo
        photo_embedding = self.face_embedder.get_embedding(image)
        
        if photo_embedding is None:
            return {"error": "No face detected in photo"}
        
        # Calculate similarity (you can use simi.py for this)
        from face.deepverify.simi import DeepSimilarityCalculator
        simi_calc = DeepSimilarityCalculator()
        similarity = simi_calc.calculate_similarity(photo_embedding, self.average_embedding)
        
        return {
            "verified": similarity > 0.6,  # Adjust threshold as needed
            "similarity": similarity,
            "embedding_dimensions": len(photo_embedding)
        }

    def clear_embeddings(self):
        """Clear all stored embeddings"""
        self.embeddings = []
        self.average_embedding = None
        self.photo_count = 0
        print("Cleared all DeepFace embeddings")

    def release(self):
        """Release resources"""
        self.face_embedder.release()


# Example usage for testing
if __name__ == "__main__":
    # Test with dummy data
    host = DeepHostEmbedding(model_name="Facenet")
    
    # Simulate photo data (in real usage, this comes from app.py)
    dummy_photos = []
    for i in range(4):
        # Create dummy photo data (in real usage, this would be actual photo bytes)
        dummy_photos.append(b"dummy_photo_data")
    
    print("Testing DeepFace host embedding processing...")
    success = host.process_photos(dummy_photos)
    
    if success:
        print("DeepFace host processing successful")
        print(f"Average embedding length: {len(host.get_average_embedding())}")
        print(f"Model info: {host.get_model_info()}")
    else:
        print("DeepFace host processing failed")
    
    host.release()
