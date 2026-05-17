# Weimo

[Poster](https://github.com/Neurotech-Davis/Weimo/tree/main/assets/Weimo_Poster.pdf) | [Presentation Slideshow](https://github.com/Neurotech-Davis/Weimo/tree/main/assets/Weimo_Conference_Slides.pdf)

## Problem

Traditional electric wheelchairs heavily rely on manual input. For the millions of people affected by neurodegenerative conditions such as ALS, joysticks are ineffective mechanisms for control. Current interfaces, such as SSVEP systems, require persistent cognitive focus and offer purely segmented control. This puts high demands on users who may struggle with precise commands.

## Approach

Our project removes the need for physical inputs by fusing three input modalities:

- **EEG motor imagery** - decodes the user's intent to move
- **LiDAR mapping** - builds a real-time 2D occupancy map of the environment
- **Eye/face tracking** - lets the user select a destination through their gaze

Together these reduce the cognitive burden on the user. Rather than requiring continuous precise commands, the system translates a user's destination and intent into autonomous locomotion.

## System Architecture

_Sensors:_

- EEG (DSI-7)
- Facecam (built-in laptop camera)
- LiDAR (RPLiDAR A1)
- Environmental Camera (Logitech C270)

_Processing:_

- EEG Classifier
- Gaze Tracker
- LiDAR environment translation
- YOLOv8 Obstacle Detection

_Actuation:_

- Raspberry Pi Pico
- Adafruit Motor HAT
- Encoder Motors
- Mecanum Wheel RC Car

## Project

In our [/src](https://github.com/Neurotech-Davis/Weimo/tree/main/src) directory, we have the code and documentation for launching the demo.
