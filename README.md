# Even-Odd Project

## Overview

Even-Odd Project is an interactive Python application that uses computer vision and hand-gesture recognition to run educational and game activities. The project currently includes a full interactive experience in `final_program.py`, along with a separate menu launcher in `main_menu.py`.

## Main Features

- Real-time hand tracking with OpenCV and MediaPipe
- Gesture-based menu navigation
- Interactive games in the GameMode folder
- Learning activities in the LearningMode folder
- Optional voice control support via `speech_recognition`
- Optional Arduino/serial output for external hardware

## Project Structure

- `final_program.py` - Main interactive app with the full game and learning flow
- `main_menu.py` - Alternate launcher/menu version
- `voice_instructions.py` - Shared voice instruction helper
- `GameMode/`
  - `game_Even_Odd.py` - Even/Odd hand-gesture game
  - `game_Rock_Paper_Scissors.py` - Rock-Paper-Scissors game
  - `train_predictor_second.py` - Model training/prediction utility
- `LearningMode/`
  - `countingMode.py` - Counting and imitation activity
  - `game_Plus_Minus.py` - Addition/subtraction learning game
  - `game_Big_Small.py` - Greater/smaller comparison game
- `models/`
  - `hand_landmarker.task` - MediaPipe hand landmark model
  - `hand_predictor.pkl` - Predictor model used by the app
  - `hand_predictor_one.pkl` - Additional predictor model file
- `aruduino_skech/` - Arduino sketch files for hardware integration

## Requirements

The project is intended to run on Python 3.x with the following libraries:

- `opencv-python`
- `mediapipe`
- `numpy`
- `speechrecognition` (optional)
- `pyserial` (optional)
- `scikit-learn` (optional, used for model loading)

## Installation

1. Install Python 3.x.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the core packages:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
4. Install optional packages if you want voice or serial support:
   ```bash
   pip install SpeechRecognition pyserial scikit-learn
   ```

## Usage

Run the main application from the project root:

```bash
python final_program.py
```

You can also try the older launcher:

```bash
python main_menu.py
```

Use your hand gestures in front of the camera to navigate the menus and play the activities.

## Notes

- Voice control is optional and only works if `speech_recognition` is installed and a microphone is available.
- Arduino/serial support is optional and only enabled when `pyserial` is installed and a compatible device is connected.
- If required dependencies are missing, the app may fall back to basic gesture-based interaction.


