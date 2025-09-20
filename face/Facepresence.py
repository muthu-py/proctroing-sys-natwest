import cv2
import mediapipe as mp
from datetime import datetime

class FacePresence:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """Process a single frame, return face count and flag status."""
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        # Count faces
        face_count = 0 if not results.detections else len(results.detections)

        # Flag logic
        flag = None
        if face_count != 1:
            flag = {
                "event": "face_presence_violation",
                "face_count": face_count,
                "timestamp": datetime.now().isoformat()
            }

        return face_count, flag

    def release(self):
        """Release resources."""
        self.face_detection.close()


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
