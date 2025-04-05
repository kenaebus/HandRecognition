import os
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Display a live camera feed with hand detection using Mediapipe.
# This code uses Mediapipe's Hand Landmarker model to detect hands via points.
# The output intends to display the detected hand points on the camera, open to AI integration for further processing hand gestures.

# Mediapipe Hand Detection - Load hand model
model_path = 'hand_landmarker.task'
if os.path.exists(model_path):
    print("Model file exists")
else:
    print("Model file does not exist")
    exit()

# Open video capture - Live camera feed
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera opened successfully")

# Create a hand landmarker instnace with the live stream mode:
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('Hand landmarks result:', result)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)
with HandLandmarker.create_from_options(options) as landmarker:

    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    
    while cap.isOpened():

        # Send live image data to perform hand landmarks detection.
        # The results are accessible via the `result_callback` provided in
        # the `HandLandmarkerOptions` object.
        # The hand landmarker must be created with the live stream mode.
         while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert BGR to RGB as Mediapipe expects RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Wrap the RGB frame into an MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Use current time as timestamp in milliseconds
            timestamp = int(time.time() * 1000)

            # Run detection
            landmarker.detect_async(mp_image, timestamp)

            # Display resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                print("Exiting...")
                break


# Display resulting frame
cap.release()
cv.destroyAllWindows()
