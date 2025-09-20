import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class SimilarityCalculator:
    def __init__(self):
        """Initialize similarity calculator"""
        pass

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding (current frame)
            embedding2: Second embedding (reference)
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            # Ensure both embeddings have the same shape
            if emb1.shape != emb2.shape:
                print(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Debug information
            print(f"Embedding 1 shape: {emb1.shape}, first 5 values: {emb1[0][:5]}")
            print(f"Embedding 2 shape: {emb2.shape}, first 5 values: {emb2[0][:5]}")
            print(f"Raw similarity: {similarity}")
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_euclidean_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Euclidean distance
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            if emb1.shape != emb2.shape:
                return float('inf')
            
            distance = np.linalg.norm(emb1 - emb2)
            return float(distance)
            
        except Exception as e:
            print(f"Error calculating Euclidean distance: {e}")
            return float('inf')

    def calculate_manhattan_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate Manhattan distance between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Manhattan distance
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            if emb1.shape != emb2.shape:
                return float('inf')
            
            distance = np.sum(np.abs(emb1 - emb2))
            return float(distance)
            
        except Exception as e:
            print(f"Error calculating Manhattan distance: {e}")
            return float('inf')

    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> dict:
        """
        Compare two embeddings using multiple similarity metrics
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Dictionary with various similarity metrics
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate various similarity metrics
            cosine_sim = self.calculate_similarity(embedding1, embedding2)
            euclidean_dist = self.calculate_euclidean_distance(embedding1, embedding2)
            manhattan_dist = self.calculate_manhattan_distance(embedding1, embedding2)
            
            # Check if embeddings are identical
            are_identical = np.array_equal(emb1, emb2)
            
            return {
                "cosine_similarity": cosine_sim,
                "euclidean_distance": euclidean_dist,
                "manhattan_distance": manhattan_dist,
                "are_identical": are_identical,
                "embedding1_sum": float(np.sum(emb1)),
                "embedding2_sum": float(np.sum(emb2)),
                "embedding1_length": len(embedding1),
                "embedding2_length": len(embedding2)
            }
            
        except Exception as e:
            return {"error": str(e)}


# Example usage for testing
if __name__ == "__main__":
    # Test similarity calculator
    simi = SimilarityCalculator()
    
    # Create dummy embeddings
    emb1 = [0.5, 0.3, 0.8, 0.2, 0.9] * 287  # 1435 dimensions
    emb2 = [0.5, 0.3, 0.8, 0.2, 0.9] * 287  # Same embedding
    
    print("Testing similarity calculator...")
    
    # Test cosine similarity
    similarity = simi.calculate_similarity(emb1, emb2)
    print(f"Cosine similarity: {similarity}")
    
    # Test comparison
    comparison = simi.compare_embeddings(emb1, emb2)
    print(f"Comparison result: {comparison}")
    
    # Test with different embeddings
    emb3 = [0.1, 0.1, 0.1, 0.1, 0.1] * 287  # Different embedding
    similarity2 = simi.calculate_similarity(emb1, emb3)
    print(f"Cosine similarity (different): {similarity2}")
