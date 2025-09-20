import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from datetime import datetime
from facematch import FaceMatcher

from deepface import DeepFace

class FacePresence:
    def __init__(self, min_detection_confidence=0.5, face_match_threshold=0.85):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face matcher
        self.face_matcher = FaceMatcher(threshold=face_match_threshold)
        self.reference_embedding_set = False
        self.photos = None

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

    def set_reference_embedding(self, embedding):
        """Set the reference embedding for face matching"""
        if embedding is not None:
            self.face_matcher.set_reference_embedding(embedding)
            self.reference_embedding_set = True
            print("Reference embedding set for face matching")

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
        elif face_count == 1 and self.reference_embedding_set:


            face_match_result = DeepFace.verify(img1_path=frame , img2_path=self.photos[0])

            # When exactly 1 face is detected, check for face matching every 10 seconds
            # match, similarity, status = self.face_matcher.match_face(frame)
            # face_match_result = {
            #     "match": match,
            #     "similarity": similarity,
            #     "status": status,
            #     "timestamp": datetime.now().isoformat()
            # }
            
            # # Only flag as violation if we actually checked and the face didn't match
            # if status == "checked" and not match:
            #     flag = {
            #         "event": "face_identity_violation",
            #         "face_count": face_count,
            #         "similarity": similarity,
            #         "timestamp": datetime.now().isoformat()
            #     }

        return face_count, flag, face_match_result

    def release(self):
        """Release resources."""
        self.face_detection.close()
        self.face_matcher.release()


# Example usage (local webcam test)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    face_monitor = FacePresence()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_count, flag = face_monitor.process_frame(frame)

        # Show face count on screen
        cv2.putText(frame, f"Faces: {face_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Presence Monitor", frame)

        if flag:
            print("⚠️ Flag raised:", flag)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_monitor.release()
    cap.release()
    cv2.destroyAllWindows()
