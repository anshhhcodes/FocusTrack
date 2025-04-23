import time
from datetime import datetime
import cv2
import dlib
import numpy as np
from PIL import Image
from imutils import face_utils
import json

from facepose_analyzer import Facepose
from utils import eye_aspect_ratio, mouth_aspect_ratio, rec_to_roi_box, crop_img, draw_axis

# Load Dlib's pretrained 68-point face landmark model
p = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    
    # Draws stylized borders with rounded corners on the face rectangle.

    # Parameters:
    # - img: image on which border is to be drawn
    # - pt1: top-left corner point (x1, y1)
    # - pt2: bottom-right corner point (x2, y2)
    # - color: color of the border
    # - thickness: thickness of the border lines
    # - r: corner radius
    # - d: line length from the corner

    x1, y1 = pt1
    x2, y2 = pt2

    # Draws each corner with two lines and a quarter-circle
    # Top-left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top-right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom-left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom-right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def main():

    # Main function to start webcam capture and run real-time face analysis
    # for blink detection, yawn detection, and head pose estimation.

    cap = cv2.VideoCapture(0)

    # Event counters and timers
    blinkCount = 0
    yawnCount = 0
    lostFocusDuration = 0
    faceNotPresentDuration = 0

    # Trackers for specific face states
    focusTimer = None
    faceTimer = None
    eyeClosed = False
    yawning = False
    lostFocus = False

    # Load head pose estimation model
    facepose = Facepose()
    shape = None
    yaw_predicted = pitch_predicted = roll_predicted = None

    # Frame rate controller
    frame_rate_use = 35
    prev = 0

    fps = 0
    fps_timer = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculating FPS (Frames Per Second)
        current_time = time.time()
        fps = 1 / (current_time - fps_timer)
        fps_timer = current_time

        # Flip and convert for visualization and detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate_use:
            prev = time.time()
            rects = detector(gray, 0)   # Detect Faces

            # If no face detected, start face-not-present timer
            if len(rects) == 0:
                if faceTimer is None:
                    faceTimer = time.time()
                faceNotPresentDuration += time.time() - faceTimer
                faceTimer = time.time()

            for rect in rects:
                faceTimer = None  # Reset face-absence timer
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Extract eyes and compute EAR (Eye Aspect Ratio)
                leftEye = shape[36:42]
                rightEye = shape[42:48]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Compute MAR (Mouth Aspect Ratio)
                mar = mouth_aspect_ratio(shape[60:69])

                # Blink detection logic
                if ear < 0.15:
                    eyeClosed = True
                if ear > 0.15 and eyeClosed:
                    blinkCount += 1
                    eyeClosed = False

                # Yawn detection logic
                if mar > 0.4:
                    yawning = True
                if mar < 0.2 and yawning:
                    yawnCount += 1
                    yawning = False

                # Extract ROI and predict head pose
                roi_box, center_x, center_y = rec_to_roi_box(rect)
                roi_img = crop_img(frame, roi_box)
                img = Image.fromarray(roi_img)
                yaw_predicted, pitch_predicted, roll_predicted = facepose.predict(img)

                # Lost focus detection based on yaw angle threshold
                if yaw_predicted.item() < -30 or yaw_predicted.item() > 30:
                    if focusTimer is None:
                        focusTimer = time.time()
                    lostFocusDuration += time.time() - focusTimer
                    focusTimer = time.time()
                    lostFocus = True
                elif lostFocus:
                    focusTimer = None

                # Create record for current frame
                record = {
                    'timestamp': datetime.now().timestamp(),
                    'yaw': yaw_predicted.item(),
                    'pitch': pitch_predicted.item(),
                    'roll': roll_predicted.item(),
                    'ear': ear,
                    'blink_count': blinkCount,
                    'mar': mar,
                    'yawn_count': yawnCount,
                    'lost_focus_duration': lostFocusDuration,
                    'face_not_present_duration': faceNotPresentDuration
                }
                print(json.dumps(record, indent=2))

            # Visualization: draw face rectangle, axis and landmarks
            if shape is not None and len(list(rects)) != 0:
                rect = list(rects)[0]
                draw_border(frame_display, (rect.left(), rect.top()),
                            (rect.left() + rect.width(), rect.top() + rect.height()), (255, 255, 255), 1, 10, 20)
                draw_axis(frame_display, yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item(),
                          tdx=int(center_x), tdy=int(center_y), size=100)

                for idx, (x, y) in enumerate(shape):
                    cv2.circle(frame_display, (x, y), 2, (255, 255, 0), -1)
                    if 36 <= idx < 48:
                        cv2.circle(frame_display, (x, y), 2, (255, 0, 255), -1)  # Eyes
                    elif 60 <= idx < 68:
                        cv2.circle(frame_display, (x, y), 2, (0, 255, 255), -1)  # Mouth

        # Display metrics on screen
        cv2.putText(frame_display, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)    # FPS display
        cv2.putText(frame_display, f"Blink Count: {blinkCount}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)   # Blink Count
        cv2.putText(frame_display, f"Yawn Count: {yawnCount}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)   # Yawn Count
        cv2.putText(frame_display, f"Lost Focus Duration: {int(lostFocusDuration)} seconds", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)   # Lost Focus Duration
        cv2.putText(frame_display, f"Face Not Present Duration: {int(faceNotPresentDuration)} seconds", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)   # Face Not Present Duration 

        cv2.imshow('Focus Track', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
