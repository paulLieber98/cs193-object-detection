## Project Goal

--> Create a simple program that uses my laptop's webcam to detect objects in real time. I'm interested in computer vision, especially how autonomous robots can recognize their surroundings.

## MVP Features
Access laptop's webcam using OpenCV.
Detect objects using a pretrained MediaPipe model.
Draw bounding boxes and labels around detected objects.

## Future Plans
Track objects as they move across frames.
Estimate the distance between objects and the camera.
Add warnings when obstacles are detected.

## Tech Stack
Python
OpenCV
Google MediaPipe

## installation + setup
git clone https://github.com/paulLieber98/cs193-object-detection
cd cs193-object-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
