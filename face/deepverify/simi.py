import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import math

class DeepSimilarityCalculator:
    def __init__(self):
        """Initialize DeepFace similarity calculator"""
        pass

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two DeepFace embeddings
        
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
            print(f"DeepFace Embedding 1 shape: {emb1.shape}, first 5 values: {emb1[0][:5]}")
            print(f"DeepFace Embedding 2 shape: {emb2.shape}, first 5 values: {emb2[0][:5]}")
            print(f"DeepFace Raw similarity: {similarity}")
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating DeepFace cosine similarity: {e}")
            return 0.0

    def calculate_euclidean_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate Euclidean distance between two DeepFace embeddings
        
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
            print(f"Error calculating DeepFace Euclidean distance: {e}")
            return float('inf')

    def calculate_manhattan_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate Manhattan distance between two DeepFace embeddings
        
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
            print(f"Error calculating DeepFace Manhattan distance: {e}")
            return float('inf')

    def calculate_dot_product_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate dot product similarity between two DeepFace embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Dot product similarity (normalized)
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            if emb1.shape != emb2.shape:
                return 0.0
            
            # Calculate dot product
            dot_product = np.dot(emb1, emb2)
            
            # Normalize by the magnitude of both vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            normalized_similarity = dot_product / (norm1 * norm2)
            return float(normalized_similarity)
            
        except Exception as e:
            print(f"Error calculating DeepFace dot product similarity: {e}")
            return 0.0

    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> Dict:
        """
        Compare two DeepFace embeddings using multiple similarity metrics
        
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
            dot_product_sim = self.calculate_dot_product_similarity(embedding1, embedding2)
            
            # Check if embeddings are identical
            are_identical = np.array_equal(emb1, emb2)
            
            # Calculate L2 norm of each embedding
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            return {
                "cosine_similarity": cosine_sim,
                "euclidean_distance": euclidean_dist,
                "manhattan_distance": manhattan_dist,
                "dot_product_similarity": dot_product_sim,
                "are_identical": are_identical,
                "embedding1_norm": float(norm1),
                "embedding2_norm": float(norm2),
                "embedding1_sum": float(np.sum(emb1)),
                "embedding2_sum": float(np.sum(emb2)),
                "embedding1_length": len(embedding1),
                "embedding2_length": len(embedding2),
                "mean_absolute_difference": float(np.mean(np.abs(emb1 - emb2)))
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_similarity_thresholds(self) -> Dict[str, float]:
        """
        Get recommended similarity thresholds for different DeepFace models
        
        Returns:
            Dictionary with model names and their recommended thresholds
        """
        return {
            "Facenet": 0.6,      # Good balance of accuracy and false positives
            "VGG-Face": 0.5,     # More lenient
            "OpenFace": 0.55,    # Moderate
            "ArcFace": 0.65,     # More strict
            "Dlib": 0.5          # More lenient
        }

    def is_similar(self, embedding1: List[float], embedding2: List[float], 
                   threshold: float = 0.6, model_name: str = "Facenet") -> Dict:
        """
        Check if two embeddings are similar based on threshold
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold (optional, uses model-specific if not provided)
            model_name: Model name for threshold selection
            
        Returns:
            Dictionary with similarity result and details
        """
        try:
            # Use model-specific threshold if not provided
            if threshold is None:
                thresholds = self.get_similarity_thresholds()
                threshold = thresholds.get(model_name, 0.6)
            
            # Calculate similarity
            similarity = self.calculate_similarity(embedding1, embedding2)
            
            # Determine if similar
            is_similar = similarity >= threshold
            
            return {
                "is_similar": is_similar,
                "similarity": similarity,
                "threshold": threshold,
                "model_name": model_name,
                "confidence": min(1.0, similarity / threshold) if threshold > 0 else 0.0
            }
            
        except Exception as e:
            return {
                "is_similar": False,
                "similarity": 0.0,
                "threshold": threshold,
                "model_name": model_name,
                "error": str(e)
            }


# Example usage for testing
if __name__ == "__main__":
    # Test similarity calculator
    simi = DeepSimilarityCalculator()
    
    # Create dummy DeepFace embeddings (typically 128-512 dimensions)
    emb1 = [0.5, 0.3, 0.8, 0.2, 0.9] * 25  # 125 dimensions (simulating Facenet)
    emb2 = [0.5, 0.3, 0.8, 0.2, 0.9] * 25  # Same embedding
    
    print("Testing DeepFace similarity calculator...")
    
    # Test cosine similarity
    similarity = simi.calculate_similarity(emb1, emb2)
    print(f"Cosine similarity: {similarity}")
    
    # Test comparison
    comparison = simi.compare_embeddings(emb1, emb2)
    print(f"Comparison result: {comparison}")
    
    # Test similarity check
    result = simi.is_similar(emb1, emb2, model_name="Facenet")
    print(f"Similarity check: {result}")
    
    # Test with different embeddings
    emb3 = [0.1, 0.1, 0.1, 0.1, 0.1] * 25  # Different embedding
    similarity2 = simi.calculate_similarity(emb1, emb3)
    print(f"Cosine similarity (different): {similarity2}")
    
    # Show recommended thresholds
    print(f"Recommended thresholds: {simi.get_similarity_thresholds()}")
