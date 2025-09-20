import numpy as np
from typing import List, Optional
from face.verify.embb import FaceEmbedding
import cv2

class HostEmbedding:
    def __init__(self):
        """Initialize host for managing multiple photo embeddings"""
        self.face_embedder = FaceEmbedding()
        self.embeddings = []
        self.average_embedding = None
        self.photo_count = 0

    def photos_returner(self , photos_data):
        images = []
        for photo_data in enumerate(photos_data):
            nparr = np.frombuffer(photo_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
            if image is None:
                print(f"Failed to decode photo {i+1}")
                continue
            images.append(image)

        return images

    def process_photos(self, photos_data: List[bytes]) -> bool:
        """
        Process multiple photos and calculate average embedding
        
        Args:
            photos_data: List of photo data as bytes
            
        Returns:
            True if successful, False otherwise
        """
        self.embeddings = []
        self.photo_count = len(photos_data)
        
        print(f"Processing {self.photo_count} photos...")
        
        for i, photo_data in enumerate(photos_data):
            print(f"Processing photo {i+1}/{self.photo_count}")
            
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(photo_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Failed to decode photo {i+1}")
                continue
            
            # Get embedding from embb.py
            embedding = self.face_embedder.get_embedding(image)
            
            if embedding is not None:
                self.embeddings.append(embedding)
                print(f"Generated embedding for photo {i+1}")
            else:
                print(f"Failed to generate embedding for photo {i+1}")
        
        if len(self.embeddings) > 0:
            self.calculate_average_embedding()
            return True
        else:
            print("No valid embeddings generated")
            return False

    def calculate_average_embedding(self):
        """Calculate average embedding from all stored embeddings"""
        if not self.embeddings:
            self.average_embedding = None
            return
        
        # Convert to numpy array for easier calculation
        embeddings_array = np.array(self.embeddings)
        
        # Calculate mean across all embeddings
        self.average_embedding = np.mean(embeddings_array, axis=0).tolist()
        
        print(f"Calculated average embedding from {len(self.embeddings)} photos")
        print(f"Average embedding shape: {len(self.average_embedding)}")
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

    def clear_embeddings(self):
        """Clear all stored embeddings"""
        self.embeddings = []
        self.average_embedding = None
        self.photo_count = 0
        print("Cleared all embeddings")

    def release(self):
        """Release resources"""
        self.face_embedder.release()


# Example usage for testing
if __name__ == "__main__":
    # Test with dummy data
    host = HostEmbedding()
    
    # Simulate photo data (in real usage, this comes from app.py)
    dummy_photos = []
    for i in range(4):
        # Create dummy photo data (in real usage, this would be actual photo bytes)
        dummy_photos.append(b"dummy_photo_data")
    
    print("Testing host embedding processing...")
    success = host.process_photos(dummy_photos)
    
    if success:
        print("Host processing successful")
        print(f"Average embedding length: {len(host.get_average_embedding())}")
    else:
        print("Host processing failed")
    
    host.release()
