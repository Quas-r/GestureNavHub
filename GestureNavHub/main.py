import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui as pg

# Screen and camera resolutions
screen_info = pg.size()
w_cam, h_cam = 1470, 956
w_scr, h_scr = screen_info.width, screen_info.height

# Mouse control variables
mouse_works = False
smoothing = 3
frame_reduction = 0

# Scroll factorso
up_scroll_factor = 15
down_scroll_factor = -2

# Time variables
prev_time = 0

# Initialize video capture
cap = cv2.VideoCapture(1)

# Previous and current mouse coordinates
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# MediaPipe initialization
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def find_landmarks(image, face_num=0):
    landmarks = []
    if results.multi_face_landmarks:
        my_face = results.multi_face_landmarks[face_num]
        for id, landmark in enumerate(my_face.landmark):
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([id, x, y])
    return landmarks

def draw_face(image, draw=True):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if draw:
                mp_draw.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
    return image

while True:
    success, image = cap.read()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = False
    image = draw_face(image)
    landmarks = find_landmarks(image)

    if len(landmarks) != 0:
        # Facial landmark indices
        nose_x, nose_y = landmarks[1][1:]
        upper_lip_x, upper_lip_y = landmarks[13][1:]
        under_lip_x, under_lip_y = landmarks[14][1:]
        left_lip_x, left_lip_y = landmarks[61][1:]
        right_lip_x, right_lip_y = landmarks[291][1:]
        left_eye_x, left_eye_y = landmarks[159][1:]
        left_eyebrow_x, left_eyebrow_y = landmarks[52][1:]
        right_eye_x, right_eye_y = landmarks[386][1:]
        right_eyebrow_x, right_eyebrow_y = landmarks[282][1:]
        left_bottom_x, left_bottom_y = landmarks[145][1:]
        right_bottom_x, right_bottom_y = landmarks[374][1:]

        # Calculate distances between facial landmarks
        lips_length = math.hypot(upper_lip_x - under_lip_x, upper_lip_y - under_lip_y)
        left_eye_brow_length = math.hypot(left_eye_x - left_eyebrow_x, left_eye_y - left_eyebrow_y)
        right_eye_brow_length = math.hypot(right_eye_x - right_eyebrow_x, right_eye_y - right_eyebrow_y)
        left_right_lips_length = math.hypot(left_lip_x - right_lip_x, left_lip_y - right_lip_y)
        left_eye_length = math.hypot(left_eye_x - left_bottom_x, left_eye_y - left_bottom_y)
        right_eye_length = math.hypot(right_eye_x - right_bottom_x, right_eye_y - right_bottom_y)

        cv2.circle(image, (nose_x, nose_y), 5, (255, 0, 255), cv2.FILLED)

        # Perform mouse movements with smoothing
        x3 = np.interp(left_eye_x, (frame_reduction, w_cam - frame_reduction), (0, w_scr))
        y3 = np.interp(left_eye_y, (frame_reduction, h_cam - frame_reduction), (0, h_scr))

        curr_x = prev_x + (x3 - prev_x) / smoothing
        curr_y = prev_y + (y3 - prev_y) / smoothing

        if mouse_works:
            pg.moveTo(w_scr - curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Scroll up when smiling and raising eyebrows
        if left_eye_brow_length > 80 and right_eye_brow_length > 80 and left_right_lips_length > 225:
            pg.scroll(up_scroll_factor)

        # Scroll down when only smiling
        elif left_right_lips_length > 225:
            pg.scroll(down_scroll_factor)
            if left_eye_length < 22:
                down_scroll_factor -= 2
                time.sleep(0.2)

        # Toggle mouse control when raising eyebrows
        elif left_eye_brow_length > 80 and right_eye_brow_length > 80:
            down_scroll_factor = -2
            mouse_works = not mouse_works
            time.sleep(0.26)

        # Blink eyes when eye length is less than a threshold
        elif left_eye_length < 22:
            pg.click()
            pg.hotkey('command', 'o')

        # Hold mouse click when mouth is opened
        if mouse_works and lips_length > 32:
            time.sleep(0.18)

    # Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    image = cv2.flip(image, 1)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Input', image)

    # Break the loop when the ESC key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
