# Even-Odd Project

## Overview

`Even-Odd Project` is an interactive Python-based learning and game application that uses computer vision to recognize hand gestures and fingers. The project includes a main menu launcher and two mode categories:

- `GameMode` for interactive games
- `LearningMode` for educational hand-gesture activities

The app is built around OpenCV and MediaPipe for real-time camera input and gesture detection.

## Features

- Hand gesture recognition using MediaPipe
- Real-time camera interface with a menu-driven experience
- Multiple games and learning activities
- Optional voice control support via `speech_recognition`
- Optional Arduino/serial integration for robot finger output

## Project Structure

- `main_menu.py` - Main launcher for the entire application
- `GameMode/`
  - `game_Even_Odd.py` - Even/Odd gesture game
  - `game_Rock_Paper_Scissors.py` - Rock-Paper-Scissors game
  - `train_predictor_second.py` - Training or predictor utility
- `LearningMode/`
  - `countingMode.py` - Finger counting mode
  - `game_Plus_Minus.py` - Math plus/minus learning game
  - `game_Big_Small.py` - Big/Small finger gesture comparison game
- `aruduino_skech/`
  - Arduino sketch files for external hardware integration

## Requirements

The project is intended to run on Python 3.x with the following libraries:

- `opencv-python`
- `mediapipe`
- `numpy`
- `speechrecognition` (optional)
- `pyserial` (optional)
- `scikit-learn` (optional; used for model loading in game mode)

## Installation

1. Install Python 3.x.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
4. Install optional packages if you want voice or serial support:
   ```bash
   pip install SpeechRecognition pyserial scikit-learn
   ```

## Usage

1. Open a terminal in the project root folder.
2. Run the main launcher:
   ```bash
   python main_menu.py
   ```
3. Use hand gestures in front of your camera to navigate the menu and play games.

## Notes

- Voice control is optional and only works if the `speech_recognition` package and a working microphone are available.
- Arduino/serial support is optional and only enabled if the `pyserial` package is installed and a compatible device is connected.
- If dependencies are missing, the application generally falls back to camera-based gesture controls.


