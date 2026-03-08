# drowsiness_detection_arduino
A real-time **Driver Drowsiness Detection System** built using **Computer Vision and Arduino-based IoT alerts**.
The system monitors a driver’s eye movements using a webcam and detects signs of fatigue. When drowsiness is detected, it triggers **multiple alert mechanisms** including LED, buzzer, vibration motor, and voice alerts.

This project demonstrates how **AI and IoT integration** can help improve road safety by preventing accidents caused by driver fatigue.

---

# Project Overview

Drowsy driving is a serious road safety issue that leads to many accidents every year. Fatigue reduces a driver's reaction time, awareness, and decision-making ability.

This project provides a **low-cost solution** that continuously monitors the driver’s eyes and detects signs of drowsiness in real time. If the driver's eyes remain closed for an extended period, the system triggers alerts to wake the driver.

The system integrates:

* Computer vision for eye detection
* Machine learning based facial landmark detection
* Arduino hardware alerts
* Real-time communication between software and hardware

---

# Features

* Real-time driver monitoring using webcam
* Eye closure detection using facial landmarks
* Eye Aspect Ratio (EAR) based drowsiness detection
* Multi-modal alert system (visual, audio, vibration, and voice)
* Integration of Python with Arduino
* Low-cost and easily reproducible hardware setup
* Real-time processing (~30 FPS)

---

# Hardware Components

| Component       | Purpose                                  |
| --------------- | ---------------------------------------- |
| Arduino UNO     | Controls the hardware alert system       |
| Breadboard      | Used for assembling the circuit          |
| Jumper Wires    | Hardware connections                     |
| Red LED         | Visual warning indicator                 |
| Active Buzzer   | Audio alert system                       |
| Vibration Motor | Haptic feedback to wake the driver       |
| MOSFET          | Used to safely drive the motor           |
| Diode           | Protects the circuit from voltage spikes |
| Resistors       | Current limiting components              |
| USB Cable       | Connects Arduino to the laptop           |

---

# Software Requirements

The project was implemented using Python and several computer vision libraries.

Required libraries:

* opencv-python
* dlib
* numpy
* scipy
* pyserial
* pyttsx3

Install dependencies using:

```
pip install opencv-python dlib-binary numpy scipy pyserial pyttsx3
```

---

# How the System Works

1. The webcam continuously captures video of the driver.
2. The computer vision model detects the driver's face in each frame.
3. Facial landmarks are extracted to locate the eyes.
4. The Eye Aspect Ratio (EAR) is calculated to determine whether the eyes are open or closed.
5. If the eyes remain closed for several consecutive frames, the system identifies drowsiness.
6. The Python program sends a command to the Arduino through serial communication.
7. The Arduino activates the alert system including LED, buzzer, and vibration motor.
8. A voice alert is also triggered to wake the driver.

---



# Applications

* Driver safety monitoring systems
* Smart vehicle systems
* Fleet driver monitoring
* Transportation safety research
* Driver assistance technologies

---

# Future Improvements

Possible future enhancements include:

* Yawn detection using mouth landmarks
* Head pose estimation
* Night-time detection using infrared cameras
* Mobile application integration
* Raspberry Pi standalone deployment
* Cloud-based monitoring systems

---

# Author

**Yusra Tariq**

---

# License

This project is intended for **educational and research purposes**.

---

