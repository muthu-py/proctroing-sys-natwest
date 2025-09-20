import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from datetime import datetime
from deepface import DeepFace
import time

class SimpleFacePresence:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Store reference photo
        self.reference_photo = None
        self.last_verification_time = 0
        self.verification_interval = 2  # 2 seconds

    def process_base64_frame(self, frame_data_json):
        """Process a base64 encoded frame from WebSocket, return face count, flag status, and face match result."""
        # Parse JSON data
        frame_data = json.loads(frame_data_json)
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame with face detection and matching
        face_count, flag, face_match_result = self.process_frame(frame)
        
        # Return response with timestamp
        response = {
            'face_count': face_count,
            'flag': flag,
            'face_match': face_match_result,
            'timestamp': frame_data.get('timestamp')
        }
        
        return response

    def set_reference_photo(self, photo_data):
        """Set the reference photo for face verification"""
        if photo_data is not None:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(photo_data, np.uint8)
            self.reference_photo = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("Reference photo set for face verification")

    def should_verify_face(self) -> bool:
        """Check if enough time has passed for the next face verification"""
        current_time = time.time()
        return (current_time - self.last_verification_time) >= self.verification_interval

    def process_frame(self, frame):
        """Process a single frame, return face count, flag status, and face match result."""
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        # Count faces
        face_count = 0 if not results.detections else len(results.detections)

        # Flag logic
        flag = None
        face_match_result = None
        
        if face_count != 1:
            flag = {
                "event": "face_presence_violation",
                "face_count": face_count,
                "timestamp": datetime.now().isoformat()
            }
        elif face_count == 1 and self.reference_photo is not None:
            # When exactly 1 face is detected, check for face verification every 2 seconds
            if self.should_verify_face():
                try:
                    # Direct DeepFace verification
                    verification_result = DeepFace.verify(
                        img1_path=frame, 
                        img2_path=self.reference_photo,
                        model_name="Facenet",
                        detector_backend="opencv",
                        distance_metric="cosine"
                    )
                    
                    # Update last verification time
                    self.last_verification_time = time.time()
                    
                    # Extract results
                    verified = verification_result["verified"]
                    distance = verification_result["distance"]
                    threshold = verification_result["threshold"]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    face_match_result = {
                        "match": verified,
                        "similarity": similarity,
                        "distance": distance,
                        "threshold": threshold,
                        "status": "checked",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Flag as violation if face doesn't match
                    if not verified:
                        flag = {
                            "event": "face_identity_violation",
                            "face_count": face_count,
                            "similarity": similarity,
                            "distance": distance,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    print(f"DeepFace verification - Verified: {verified}, Similarity: {similarity:.4f}, Distance: {distance:.4f}")
                    
                except Exception as e:
                    print(f"DeepFace verification error: {e}")
                    face_match_result = {
                        "match": False,
                        "similarity": 0.0,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Flag as violation on error
                    flag = {
                        "event": "face_verification_error",
                        "face_count": face_count,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                # Not time to check yet
                face_match_result = {
                    "match": True,
                    "similarity": 1.0,
                    "status": "not_checked",
                    "timestamp": datetime.now().isoformat()
                }

        return face_count, flag, face_match_result

    def get_verification_info(self):
        """Get verification information"""
        return {
            "reference_photo_set": self.reference_photo is not None,
            "last_verification_time": self.last_verification_time,
            "verification_interval": self.verification_interval,
            "next_verification_in": max(0, self.verification_interval - (time.time() - self.last_verification_time))
        }

    def release(self):
        """Release resources."""
        self.face_detection.close()


# Example usage (local webcam test)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    face_monitor = SimpleFacePresence()

    # Load a reference photo (you would get this from app.py)
    # For testing, we'll use a dummy photo
    print("Please set a reference photo first")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_count, flag, face_match_result = face_monitor.process_frame(frame)

        # Show face count on screen
        cv2.putText(frame, f"Faces: {face_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show face match info if available
        if face_match_result:
            match_text = f"Match: {face_match_result['match']}"
            sim_text = f"Sim: {face_match_result['similarity']:.3f}"
            status_text = f"Status: {face_match_result['status']}"
            
            cv2.putText(frame, match_text, (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, sim_text, (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Simple Face Presence Monitor", frame)

        if flag:
            print("⚠️ Flag raised:", flag)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_monitor.release()
    cap.release()
    cv2.destroyAllWindows()
