import time
import os
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import math
import numpy as np
import pyautogui


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

dir_path = os.path.dirname(__file__)
MODEL_PATH = os.path.join(dir_path, "models", "hand_landmarker.task")

def extract_landmarks(hand_landmarks_list):
    landmarks = []
    for hand_landmarks in hand_landmarks_list:
        for landmark in hand_landmarks:
            landmarks.append((landmark.x, landmark.y, landmark.z))
    return landmarks


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow("Frame", img)


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Draw landmark indices/numbers on the image.
        for i, landmark in enumerate(hand_landmarks):
            h, w, _ = annotated_image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.putText(
                annotated_image,
                str(i),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 0, 0),  # Color: Blue
                1,  # Thickness
                cv2.LINE_AA
            )

    return annotated_image




# Options for the hand landmarker
base_options = BaseOptions(
    model_asset_path=MODEL_PATH
)  # Ensure this path is correct and points to a .tflite file
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=RunningMode.IMAGE,
)
detector = HandLandmarker.create_from_options(options)

# Setup camera capture
cap = cv2.VideoCapture(0)

# Get the number of frames
NUM_FRAMES = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
print(f"Number of frames in the video {NUM_FRAMES}")

if not cap.isOpened():
    print("Failed to open capture device")
    exit(1)

# Run inference on the video
print("Running hand landmarker...")

frame_count = 0

x_screen, y_screen = pyautogui.size()

pointer1 = [0,0]
# while True:
while True:
    frame_count += 1
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using HandLandmarker
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    
    # Draw the hand landmarks on the frame
    if results:
        landmarks = extract_landmarks(results.hand_landmarks)
        
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
        hand_landmarks_list = results.hand_landmarks
        try:
            pointer = landmarks[8]
            pouce = landmarks[4]
            x = pointer[0]*x_screen
            y = pointer[1]*y_screen+100
            
            print(abs(pointer[1] - pouce[1])*100 ,"  ", abs(pointer[0] - pouce[0])*100)
            if abs(pointer[1] - pouce[1])*100 < 5 and abs(pointer[0] - pouce[0])*100 < 1:
                pyautogui.click()
                
            pyautogui.moveTo(x, y)
            pointer1 = pointer
        except Exception as e :
            pass
    # For local use
    cv2.imshow("Frame", annotated_image)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()
