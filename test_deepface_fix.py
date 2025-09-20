#!/usr/bin/env python3
"""
Test script to verify DeepFace verification fixes
"""

import cv2
import numpy as np
from face.Facepresence_simple import SimpleFacePresence
from face.deepverify.embb import DeepFaceEmbedding

def test_simple_face_presence():
    """Test SimpleFacePresence with proper error handling"""
    print("Testing SimpleFacePresence...")
    
    # Create a test instance
    face_presence = SimpleFacePresence()
    
    # Create a dummy reference photo (random image)
    dummy_photo = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Set reference photo
    face_presence.set_reference_photo(dummy_photo.tobytes())
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test process_frame
    try:
        face_count, flag, face_match_result = face_presence.process_frame(test_frame)
        print(f"✓ SimpleFacePresence test passed - Face count: {face_count}")
        print(f"  Flag: {flag}")
        print(f"  Face match result: {face_match_result}")
        return True
    except Exception as e:
        print(f"✗ SimpleFacePresence test failed: {e}")
        return False

def test_deepface_embedding():
    """Test DeepFaceEmbedding with proper error handling"""
    print("\nTesting DeepFaceEmbedding...")
    
    # Create a test instance
    embedder = DeepFaceEmbedding()
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test get_embedding
    try:
        embedding = embedder.get_embedding(test_frame)
        if embedding is not None:
            print(f"✓ DeepFaceEmbedding test passed - Embedding length: {len(embedding)}")
        else:
            print("✓ DeepFaceEmbedding test passed - No face detected (expected)")
        return True
    except Exception as e:
        print(f"✗ DeepFaceEmbedding test failed: {e}")
        return False
    
    # Test verify_faces
    try:
        test_frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = embedder.verify_faces(test_frame, test_frame2)
        print(f"✓ DeepFaceEmbedding verify_faces test passed - Result: {result}")
        return True
    except Exception as e:
        print(f"✗ DeepFaceEmbedding verify_faces test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running DeepFace verification fix tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test SimpleFacePresence
    if test_simple_face_presence():
        tests_passed += 1
    
    # Test DeepFaceEmbedding
    if test_deepface_embedding():
        tests_passed += 1
    
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! DeepFace verification errors should be fixed.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()
