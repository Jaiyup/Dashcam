# basic imports
import cv2
import dlib
import pygame
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

# eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# mouth aspect ratio
def mouth_aspect_ratio(mouth):
    X = dist.euclidean(mouth[0], mouth[6])
    Y1 = dist.euclidean(mouth[2], mouth[10])
    Y2 = dist.euclidean(mouth[4], mouth[8])
    return ((Y1 + Y2) / 2.0) / X


# sound
pygame.mixer.init(22050, -16, 2, 4096)
pygame.mixer.music.load("tone1.mp3")

# camera
camera = cv2.VideoCapture(0)

# video output
frame_width = 640
frame_height = 400
out = cv2.VideoWriter(
    "Output.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (frame_width, frame_height)
)

# face detector and landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# thresholds
EYE_AR_CONSEC_FRAMES = 48
MOU_AR_THRESH = 0.75

COUNTER = 0
blink_count = 0
blink_status = False

yawn_status = False
yawn_count = 0

user_ear = []

# main loop
while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev_blink = blink_status
    prev_yawn = yawn_status

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        user_ear.append(ear)

        # draw face parts
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        # eye check
        if ear < user_ear[0]:
            COUNTER += 1
            blink_status = True
            cv2.putText(frame, "Eyes Closed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.play()
        else:
            COUNTER = 0
            blink_status = False
            pygame.mixer.music.stop()
            cv2.putText(frame, "Eyes Open", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if prev_blink and not blink_status:
            blink_count += 1

        # yawn check
        if mar > MOU_AR_THRESH:
            yawn_status = True
            cv2.putText(frame, "Yawning", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            yawn_status = False

        if prev_yawn and not yawn_status:
            yawn_count += 1

        # text info
        cv2.putText(frame, f"Blink Count: {blink_count}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Senior Design Project", (350, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (153, 51, 102), 1)

    out.write(frame)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
out.release()
cv2.destroyAllWindows()
