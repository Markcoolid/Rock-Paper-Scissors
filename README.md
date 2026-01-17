# Rock Paper Scissors
## Overview
This is a small machine learning project that plays Rock Paper Scissors using a webcam. The system detects the player's hand gesture and will generate its own random hand to play and compare against.

## Why I Built This
I built this project to explore computer vision and real-time gesture recognition using YOLO26.

## How It Works
- The model is based on **Ultralytics YOLO26** and trained on a dataset of Rock Paper Scissors gestures from Kaggle.
- The runtime code uses **OpenCV** to capture webcam input and classify gestures in real time.
- The program compares the player’s gesture with its own choice and determines the winner.

The code is organized into:
- `/training` - code and scripts used to train the model  
- `/runtime` - code used to run the game with webcam input  

## How to Run
Requirements:
- Python 3.x
- Webcam
- OpenCV (`pip install opencv-python`)  
- Ultralytics YOLO (`pip install ultralytics`)

Run the game:
```bash
cd runtime
python main.py
