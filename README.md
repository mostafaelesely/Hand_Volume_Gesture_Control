# Hand Gesture Volume Control

This project demonstrates a hand gesture-based volume control system using a webcam. It leverages OpenCV for image processing, MediaPipe for hand tracking, and Pycaw for controlling the system audio. The program captures video from the webcam, detects hand gestures, and adjusts the system volume based on the distance between the thumb and index finger.

## Features

- **Hand Detection**: Utilizes MediaPipe to detect and track hands in real-time.
- **Volume Control**: Adjusts the system volume based on the distance between the thumb and index finger.
- **Real-time Processing**: Processes video feed from the webcam in real-time to detect hand gestures and adjust volume dynamically.

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Pycaw
- Comtypes

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hand-gesture-volume-control.git
    cd hand-gesture-volume-control
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python mediapipe numpy pycaw comtypes
    ```

## Usage

Run the script to start the hand gesture volume control:
```bash
python hand_gesture_volume_control.py
```

### How It Works

1. **Setup**: Initializes the webcam and sets up MediaPipe for hand detection.
2. **Volume Control Initialization**: Uses Pycaw to interface with the system audio and retrieve the volume range.
3. **Hand Detection and Landmark Extraction**: Captures frames from the webcam, processes them with MediaPipe to detect hand landmarks.
4. **Gesture Recognition**: Identifies the thumb and index finger positions and calculates the distance between them.
5. **Volume Adjustment**: Maps the distance between fingers to the system volume range and adjusts the volume accordingly.
6. **Visualization**: Displays the video feed with hand landmarks, a volume bar, and the current volume percentage.

### Key Functions and Variables

- **mp_drawing**: Utilities for drawing landmarks.
- **mp_hands**: MediaPipe's hand tracking model.
- **volume**: Interface for controlling the system volume.
- **volRange**: Range of system volume levels.
- **minVol, maxVol**: Minimum and maximum volume levels.
- **volBar, volPer**: Variables for displaying volume bar and percentage.
- **lmList**: List of landmarks detected on the hand.
- **x1, y1, x2, y2**: Coordinates of the thumb and index finger.
- **length**: Distance between the thumb and index finger used for volume control.

### Code Explanation

The script follows these steps:

1. **Initialization**: Sets up webcam, MediaPipe hands model, and volume control interface.
2. **Main Loop**:
   - Captures frame from webcam.
   - Converts frame to RGB and processes it with MediaPipe to detect hands.
   - Extracts landmarks and calculates distance between thumb and index finger.
   - Adjusts volume based on the calculated distance.
   - Displays the annotated frame with landmarks, volume bar, and percentage.
3. **Termination**: Releases the webcam and closes all OpenCV windows when 'q' is pressed.

## Troubleshooting

- Ensure your webcam is properly connected.
- Install all required dependencies.
- Adjust the detection confidence parameters if the hand detection is not accurate.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

