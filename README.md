# Step By Hand

## Overview

Step By Hand is an interactive Python project that uses computer vision and hand-gesture recognition to power a small educational experience. The main application is implemented in `final_program.py`.
YouTube Link: https://youtu.be/rZDtoZ4lcIM

## Main Features

- Real-time hand tracking with OpenCV and MediaPipe
- A main menu with two modes: Game Mode and Learning Mode
- Gesture-based navigation and interaction
- Optional voice control support via `speech_recognition`
- Optional Arduino/serial output for external hardware

## Project Structure

The current workspace contains the following files and folders:

- `final_program.py` - Main application entry point. It contains the full interface logic for the menu system, Game Mode, and Learning Mode.
- `voice_instructions.py` - Shared helper for voice prompts and speech output.
- `models/` - Contains the hand landmark model and predictor files used by the app.
  - `hand_landmarker.task`
  - `hand_predictor.pkl`
  - `hand_predictor_one.pkl`
- `aruduino_skech/` - Arduino sketch folder for optional hardware integration.
  - `aruduino_skech.ino`

### Modes in the app

- Game Mode
  - Rock Paper Scissors
  - Even/Odd
- Learning Mode
  - Counting / Imitation
  - Plus / Minus
  - Greater / Smaller

> The current repository does not contain separate `GameMode/` or `LearningMode/` folders. Those activities are implemented inside `final_program.py` as different screens and states.

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

Use your hand gestures in front of the camera to navigate the menus and play the activities.

## Notes

- Voice control is optional and only works if `speech_recognition` is installed and a microphone is available.
- Arduino/serial support is optional and only enabled when `pyserial` is installed and a compatible device is connected.
- If required dependencies are missing, the app may fall back to basic gesture-based interaction.


