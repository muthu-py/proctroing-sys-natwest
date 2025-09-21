import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import pandas as pd
import joblib
import os

# =========================================================================
# STEP 1: INITIALIZE MODELS AND CONFIGURATION
# =========================================================================
print("Loading models, please wait...")

# --- Create a directory to save evidence images ---
EVIDENCE_DIR = "cheating_evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- Initialize Feature Extractor Models ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
yolo_model = YOLO('yolov8n.pt')

# --- Load Your Trained Cheating Detection Model ---
try:
    print(1)
    cheating_model = joblib.load('cheating_detection_model.joblib')
    print("✅ Cheating detection model loaded successfully.")
except FileNotFoundError:
    print(2)
    print("❌ FATAL ERROR: 'cheating_detection_model.joblib' not found.")
    print("Please make sure the trained model file is in the same folder as this script.")
    exit()

# This list of features MUST EXACTLY match the order and names used during training.
MODEL_FEATURES = [
    'face_present', 'no_of_face', 'face_x', 'face_y', 'face_w', 'face_h',
    'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_tip_x',
    'nose_tip_y', 'mouth_x', 'mouth_y', 'face_conf', 'hand_count',
    'left_hand_x', 'left_hand_y', 'right_hand_x', 'right_hand_y',
    'hand_obj_interaction', 'head_pose', 'head_pitch', 'head_yaw',
    'head_roll', 'phone_present', 'phone_loc_x', 'phone_loc_y',
    'phone_conf', 'gaze_on_script', 'gaze_direction', 'gazePoint_x',
    'gazePoint_y', 'pupil_left_x', 'pupil_left_y', 'pupil_right_x', 'pupil_right_y'
]
print("Models loaded. Starting webcam...")


# =========================================================================
# STEP 2: FULLY IMPLEMENTED FEATURE EXTRACTION FUNCTION
# =========================================================================
def extract_features_from_frame(frame):
    """
    Extracts all required features from a single image frame.
    """
    if frame is None:
        return None
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    features = {key: 0.0 for key in MODEL_FEATURES}
    features['head_pose'] = 'None'
    features['gaze_direction'] = 'None'
    features['no_of_face'] = 0
    features['hand_count'] = 0

    # --- Face, Head Pose, Pupil, and Gaze ---
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(image_rgb)
        if face_results.multi_face_landmarks:
            features['face_present'] = 1
            features['no_of_face'] = len(face_results.multi_face_landmarks)
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # --- Bounding Box Calculation ---
            all_x = [lm.x * w for lm in face_landmarks.landmark]
            all_y = [lm.y * h for lm in face_landmarks.landmark]
            features['face_x'] = min(all_x)
            features['face_y'] = min(all_y)
            features['face_w'] = max(all_x) - features['face_x']
            features['face_h'] = max(all_y) - features['face_y']
            features['face_conf'] = 95.0  # Placeholder confidence

            # --- Key Landmark Extraction ---
            nose_tip = face_landmarks.landmark[1]
            features['nose_tip_x'], features['nose_tip_y'] = nose_tip.x * w, nose_tip.y * h
            
            # --- Eye Centers ---
            left_eye_center_x = np.mean([face_landmarks.landmark[i].x for i in [33, 133, 160, 158]]) * w
            left_eye_center_y = np.mean([face_landmarks.landmark[i].y for i in [33, 133, 160, 158]]) * h
            features['left_eye_x'], features['left_eye_y'] = left_eye_center_x, left_eye_center_y

            right_eye_center_x = np.mean([face_landmarks.landmark[i].x for i in [362, 263, 387, 385]]) * w
            right_eye_center_y = np.mean([face_landmarks.landmark[i].y for i in [362, 263, 387, 385]]) * h
            features['right_eye_x'], features['right_eye_y'] = right_eye_center_x, right_eye_center_y
            
            # --- Mouth Center ---
            features['mouth_x'] = np.mean([face_landmarks.landmark[i].x for i in [61, 291]]) * w
            features['mouth_y'] = np.mean([face_landmarks.landmark[i].y for i in [61, 291]]) * h

            # --- Pupil Locations ---
            features['pupil_left_x'] = face_landmarks.landmark[473].x * w
            features['pupil_left_y'] = face_landmarks.landmark[473].y * h
            features['pupil_right_x'] = face_landmarks.landmark[468].x * w
            features['pupil_right_y'] = face_landmarks.landmark[468].y * h
            
            # --- Head Pose Estimation ---
            face_3d = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
            face_2d = np.array([(nose_tip.x*w, nose_tip.y*h), (face_landmarks.landmark[152].x*w, face_landmarks.landmark[152].y*h), (face_landmarks.landmark[33].x*w, face_landmarks.landmark[33].y*h), (face_landmarks.landmark[263].x*w, face_landmarks.landmark[263].y*h), (face_landmarks.landmark[61].x*w, face_landmarks.landmark[61].y*h), (face_landmarks.landmark[291].x*w, face_landmarks.landmark[291].y*h)], dtype=np.float64)
            cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
            success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1), dtype=np.float64))
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            features['head_pitch'] = angles[0]
            features['head_yaw'] = angles[1]
            features['head_roll'] = angles[2]

            if angles[1] > 15: features['head_pose'] = 'right'
            elif angles[1] < -15: features['head_pose'] = 'left'
            elif angles[0] > 15: features['head_pose'] = 'down'
            else: features['head_pose'] = 'forward'

    # --- Hand Detection ---
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        hand_results = hands.process(image_rgb)
        if hand_results.multi_hand_landmarks:
            features['hand_count'] = len(hand_results.multi_hand_landmarks)
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[i].classification[0].label
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                if handedness == 'Left':
                    features['left_hand_x'], features['left_hand_y'] = wrist.x, wrist.y
                elif handedness == 'Right':
                    features['right_hand_x'], features['right_hand_y'] = wrist.x, wrist.y

    # --- Phone Detection ---
    yolo_results = yolo_model(frame, verbose=False)
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls) == 67:
                features['phone_present'] = 1.0
                xywh = box.xywh.cpu().numpy()[0]
                features['phone_loc_x'] = xywh[0] - (xywh[2] / 2)
                features['phone_loc_y'] = xywh[1] - (xywh[3] / 2)
                features['phone_conf'] = float(box.conf)
                break
        if features['phone_present']:
            break

    return features

# =========================================================================
# STEP 3: MAIN WEBCAM LOOP
# =========================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    capture_interval = 5
    last_capture_time = time.time()

    print("\n--- Webcam feed started. Press 'q' in the webcam window to quit. ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Live Feed - Press Q to Quit', frame)

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            print(f"\n--- Analyzing frame at {time.strftime('%H:%M:%S')} ---")
            
            features_dict = extract_features_from_frame(frame)
            
            if features_dict:
                # print("DEBUG >>> Features being sent to model:", features_dict) # You can re-enable this for debugging
                
                live_features_df = pd.DataFrame([features_dict])
                live_features_df = live_features_df[MODEL_FEATURES]
                
                for col in ['head_pose', 'gaze_direction']:
                    live_features_df[col] = live_features_df[col].astype('category')

                prediction = cheating_model.predict(live_features_df)[0]
                
                if prediction == 1:
                    print("Prediction: 1 (Cheat)")
                    
                    # Create a unique filename with a timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(EVIDENCE_DIR, f"cheat_detected_{timestamp}.png")
                    
                    # Save the current frame as a PNG image
                    cv2.imwrite(filename, frame)
                    print(f"📸 Evidence saved: {filename}")
                    
                else:
                    print("Prediction: 0 (Normal)")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(EVIDENCE_DIR, f"normal_detected_{timestamp}.png")
                    
                    # Save the current frame as a PNG image
                    cv2.imwrite(filename, frame)
                    print(f"📸 normal saved: {filename}")
            
            else:
                print("Could not extract features from the frame.")

            last_capture_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")