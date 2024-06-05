import cv2  # Importing OpenCV library for image and video processing.
import mediapipe as mp  # Importing Mediapipe library for hand tracking.
import math  # Importing math module for mathematical operations.
import numpy as np  # Importing NumPy library for numerical operations.
from ctypes import cast, POINTER  # Importing necessary functions from ctypes module.
from comtypes import CLSCTX_ALL  # Importing necessary constant from comtypes module.
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # Importing necessary functions from pycaw library.
#from cvzone.HandTrackingModule import HandDetector  # Importing HandDetector class from cvzone library.

# Mediapipe solution APIs for hand detection and drawing landmarks.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Library Usage - Getting audio devices and volume interface.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # Getting the range of volume control.
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0  # Initializing volume control variables.

# Webcam Setup - Setting up webcam for capturing video.
wCam, hCam = 1920, 1080
cam = cv2.VideoCapture(0)
cam.set(3, wCam)  # Setting width of the video feed.
cam.set(4, hCam)  # Setting height of the video feed.
#detector = HandDetector(detectionCon=0.8, maxHands=2)  # Initializing HandDetector object.

# Using Mediapipe Hand Landmark Model for hand tracking.
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.2) as hands:
    while cam.isOpened():  # Loop for capturing frames from webcam.
        success, image = cam.read()  # Reading a frame from the webcam feed.
        image = cv2.flip(image, 1)  # Flipping the frame horizontally.y3ni by5ly my right hand tzhr zy ma hya fe elsora



        # Converting image to RGB format for Mediapipe processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processing hand landmarks using Mediapipe.
        results = hands.process(image)

        # Converting image back to BGR format for displaying with OpenCV.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Drawing hand landmarks and connections if hands are detected.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Extracting landmark positions from the detected hands.
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        # Extracting Thumb and Index finger positions if hand landmarks are detected.
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Marking Thumb and Index finger with circles and connecting them with a line.
            cv2.circle(image, (x1, y1), 15, (255, 255, 0))
            cv2.circle(image, (x2, y2), 15, (255, 255, 0))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculating distance between Thumb and Index finger for volume control.
            length = math.hypot(x2 - x1, y2 - y1)

            # Interpolating volume based on finger distance and updating volume level.
            vol = np.interp(length, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Interpolating volume bar position and volume percentage based on finger distance.
            volBar = np.interp(length, [50, 220], [400, 150])
            volPer = np.interp(length, [50, 220], [0, 100])

            # Drawing volume bar and percentage on the frame.
            cv2.rectangle(image, (50, 150), (85, 400), (60, 50, 120), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (80, 90, 150), cv2.FILLED)
            cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (75, 30, 200), 3)

        # Displaying the annotated frame with OpenCV.
        cv2.imshow('handDetector', image)

        # Exiting the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Releasing the webcam and closing OpenCV windows.
cam.release()
