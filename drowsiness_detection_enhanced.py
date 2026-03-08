"""
Enhanced Driver Drowsiness Detection System v2.0
Additional Features: Data Logging, Performance Monitoring, Multi-level Alerts
"""

import cv2
import numpy as np
import dlib
import pyttsx3
import serial
import time
import threading
from scipy.spatial import distance
from collections import deque
import logging
import csv
from datetime import datetime
import json


class EnhancedDrowsinessDetector:
    """Enhanced version with logging, performance monitoring, and multi-level alerts"""
    
    LEFT_EYE_INDICES = list(range(36, 42))
    RIGHT_EYE_INDICES = list(range(42, 48))
    MOUTH_INDICES = list(range(48, 68))
    
    # Multi-level alert thresholds
    WARNING_EAR = 0.27      # Warning level
    DROWSY_EAR = 0.25       # Drowsy level
    CRITICAL_EAR = 0.20     # Critical level
    
    WARNING_FRAMES = 30     # ~1 second
    DROWSY_FRAMES = 48      # ~2 seconds
    CRITICAL_FRAMES = 90    # ~3 seconds
    
    def __init__(self, arduino_port='COM3', driver_name='Yusra', 
                 use_arduino=True, enable_logging=True):
        """
        Initialize enhanced detector
        
        Args:
            arduino_port (str): Serial port for Arduino
            driver_name (str): Driver name for personalized alerts
            use_arduino (bool): Whether to use Arduino integration
            enable_logging (bool): Whether to enable data logging
        """
        self.driver_name = driver_name
        self.use_arduino = use_arduino
        self.enable_logging = enable_logging
        
        # Setup logging
        self.setup_logging()
        
        # Initialize core components
        logger.info("Initializing enhanced drowsiness detector...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 1.0)
        
        # Initialize Arduino
        self.arduino = None
        if self.use_arduino:
            self.init_arduino(arduino_port)
        
        # State tracking
        self.ear_history = deque(maxlen=100)
        self.drowsy_frames = 0
        self.alert_level = 0  # 0=normal, 1=warning, 2=drowsy, 3=critical
        self.last_alert_time = 0
        self.blink_start_time = None
        
        # Statistics
        self.stats = {
            'total_blinks': 0,
            'warning_events': 0,
            'drowsy_events': 0,
            'critical_events': 0,
            'total_frames': 0,
            'session_start': datetime.now(),
            'avg_ear': 0.0,
            'min_ear': 1.0,
            'max_ear': 0.0
        }
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Data logging
        if self.enable_logging:
            self.init_logging_file()
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        log_format = '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'drowsiness_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def init_logging_file(self):
        """Initialize CSV logging for data analysis"""
        self.log_filename = f'drowsiness_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'ear', 'left_ear', 'right_ear', 
                'alert_level', 'drowsy_frames', 'blinks', 
                'head_tilt', 'fps'
            ])
        
        logger.info(f"Data logging initialized: {self.log_filename}")
    
    def log_data(self, ear, left_ear, right_ear, head_tilt, fps):
        """Log detection data to CSV"""
        if not self.enable_logging:
            return
        
        try:
            with open(self.log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    f"{ear:.4f}",
                    f"{left_ear:.4f}",
                    f"{right_ear:.4f}",
                    self.alert_level,
                    self.drowsy_frames,
                    self.stats['total_blinks'],
                    f"{head_tilt:.2f}",
                    f"{fps:.2f}"
                ])
        except Exception as e:
            logger.error(f"Error logging data: {e}")
    
    def init_arduino(self, port):
        """Initialize Arduino with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Arduino on {port} (attempt {attempt+1}/{max_retries})...")
                self.arduino = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                
                # Verify connection
                self.arduino.write(b"STATUS\n")
                response = self.arduino.readline().decode().strip()
                
                if "SYSTEM STATUS" in response or response:
                    logger.info("Arduino connected and verified")
                    return
                    
            except serial.SerialException as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error("Failed to connect to Arduino after all retries")
                    self.use_arduino = False
    
    @staticmethod
    def calculate_ear(eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio for yawn detection
        Future enhancement for detecting yawning
        """
        vertical_1 = distance.euclidean(mouth_landmarks[13], mouth_landmarks[19])
        vertical_2 = distance.euclidean(mouth_landmarks[14], mouth_landmarks[18])
        vertical_3 = distance.euclidean(mouth_landmarks[15], mouth_landmarks[17])
        horizontal = distance.euclidean(mouth_landmarks[12], mouth_landmarks[16])
        
        mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
        return mar
    
    def get_head_tilt_angle(self, landmarks):
        """Calculate head tilt angle"""
        left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) 
                           for i in self.LEFT_EYE_INDICES], axis=0)
        right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) 
                            for i in self.RIGHT_EYE_INDICES], axis=0)
        
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        
        return abs(angle)
    
    def extract_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks"""
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                        for i in eye_indices])
    
    def extract_mouth_landmarks(self, landmarks):
        """Extract mouth landmarks"""
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                        for i in self.MOUTH_INDICES])
    
    def send_arduino_command(self, command):
        """Send command to Arduino with error handling"""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(f"{command}\n".encode())
                logger.debug(f"Arduino: {command}")
                return True
            except serial.SerialException as e:
                logger.error(f"Arduino communication error: {e}")
                return False
        return False
    
    def play_voice_alert(self, message):
        """Play voice alert with threading"""
        def speak():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def determine_alert_level(self, ear, frames):
        """
        Determine alert level based on EAR and frame count
        Returns: 0=normal, 1=warning, 2=drowsy, 3=critical
        """
        if ear >= self.WARNING_EAR:
            return 0  # Normal
        elif ear >= self.DROWSY_EAR and frames >= self.WARNING_FRAMES:
            return 1  # Warning
        elif ear >= self.CRITICAL_EAR and frames >= self.DROWSY_FRAMES:
            return 2  # Drowsy
        elif frames >= self.CRITICAL_FRAMES:
            return 3  # Critical
        return 0
    
    def trigger_alert(self, level):
        """
        Trigger multi-level alert
        Level 0: Normal (no alert)
        Level 1: Warning (visual only)
        Level 2: Drowsy (voice + visual)
        Level 3: Critical (voice + audio + visual + haptic)
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_alert_time < 3:
            return
        
        self.last_alert_time = current_time
        self.alert_level = level
        
        if level == 1:  # Warning
            logger.warning("WARNING: Reduced eye opening detected")
            self.stats['warning_events'] += 1
            if self.use_arduino:
                self.send_arduino_command('LED_ON')
                threading.Timer(2.0, lambda: self.send_arduino_command('LED_OFF')).start()
        
        elif level == 2:  # Drowsy
            logger.warning("DROWSY: Driver showing signs of drowsiness")
            self.stats['drowsy_events'] += 1
            self.play_voice_alert(f"{self.driver_name}, please stay alert")
            if self.use_arduino:
                self.send_arduino_command('ALERT_ON')
                threading.Timer(2.0, lambda: self.send_arduino_command('ALERT_OFF')).start()
        
        elif level == 3:  # Critical
            logger.error("CRITICAL: Severe drowsiness detected!")
            self.stats['critical_events'] += 1
            self.play_voice_alert(f"{self.driver_name}, wake up immediately! Pull over safely!")
            if self.use_arduino:
                self.send_arduino_command('ALERT_ON')
                threading.Timer(4.0, lambda: self.send_arduino_command('ALERT_OFF')).start()
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_history.append(fps)
            self.last_fps_time = current_time
            return np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        return 0
    
    def update_statistics(self, ear):
        """Update running statistics"""
        self.stats['total_frames'] += 1
        self.stats['avg_ear'] = (self.stats['avg_ear'] * (self.stats['total_frames'] - 1) + ear) / self.stats['total_frames']
        self.stats['min_ear'] = min(self.stats['min_ear'], ear)
        self.stats['max_ear'] = max(self.stats['max_ear'], ear)
    
    def draw_enhanced_status(self, frame, ear, left_ear, right_ear, head_tilt, fps):
        """Draw enhanced status information"""
        # Background panel
        cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
        
        # Alert level color
        level_colors = [
            (0, 255, 0),    # Normal - Green
            (0, 255, 255),  # Warning - Yellow
            (0, 165, 255),  # Drowsy - Orange
            (0, 0, 255)     # Critical - Red
        ]
        color = level_colors[self.alert_level]
        
        # EAR values
        cv2.putText(frame, f"EAR: {ear:.3f} (L:{left_ear:.3f} R:{right_ear:.3f})", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status
        status_text = ['AWAKE', 'WARNING', 'DROWSY', 'CRITICAL'][self.alert_level]
        cv2.putText(frame, f"Status: {status_text}", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Frame counter with threshold bars
        threshold_text = f"{self.drowsy_frames}/{self.DROWSY_FRAMES}"
        cv2.putText(frame, f"Drowsy Frames: {threshold_text}", (15, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics
        cv2.putText(frame, f"Blinks: {self.stats['total_blinks']} | "
                          f"W:{self.stats['warning_events']} "
                          f"D:{self.stats['drowsy_events']} "
                          f"C:{self.stats['critical_events']}", 
                   (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance
        cv2.putText(frame, f"FPS: {fps:.1f} | Frames: {self.stats['total_frames']}", 
                   (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Head tilt
        if head_tilt > 0:
            cv2.putText(frame, f"Head Tilt: {head_tilt:.1f}°", (15, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw EAR history graph
        self.draw_ear_graph(frame, 460, 10, 180, 100)
    
    def draw_ear_graph(self, frame, x, y, width, height):
        """Draw real-time EAR graph"""
        if len(self.ear_history) < 2:
            return
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Threshold lines
        threshold_y = int(y + height - (self.DROWSY_EAR / 0.5) * height)
        cv2.line(frame, (x, threshold_y), (x + width, threshold_y), (0, 0, 255), 1)
        
        # EAR history line
        points = []
        for i, ear in enumerate(self.ear_history):
            px = x + int((i / len(self.ear_history)) * width)
            py = y + height - int((ear / 0.5) * height)
            py = max(y, min(y + height, py))
            points.append((px, py))
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
        
        # Label
        cv2.putText(frame, "EAR History", (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Process frame with enhanced features"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("uint8")
        faces = self.detector(gray, 0)
        
        ear = 0.0
        left_ear = 0.0
        right_ear = 0.0
        head_tilt = 0.0
        fps = self.calculate_fps()
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = self.predictor(gray, face)
            
            # Extract eye landmarks
            left_eye = self.extract_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
            right_eye = self.extract_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
            
            # Calculate EAR
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate head tilt
            head_tilt = self.get_head_tilt_angle(landmarks)
            
            # Draw eye contours
            for eye_points in [left_eye, right_eye]:
                eye_hull = cv2.convexHull(eye_points.astype(np.int32))
                cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
            
            # Add to history
            self.ear_history.append(ear)
            
            # Update statistics
            self.update_statistics(ear)
            
            # Drowsiness detection
            if ear < self.WARNING_EAR:
                self.drowsy_frames += 1
                
                if self.blink_start_time is None:
                    self.blink_start_time = time.time()
                
                # Determine alert level
                level = self.determine_alert_level(ear, self.drowsy_frames)
                if level > self.alert_level:
                    self.trigger_alert(level)
            else:
                # Reset
                if self.drowsy_frames > 5:
                    self.stats['total_blinks'] += 1
                
                self.drowsy_frames = 0
                self.blink_start_time = None
                self.alert_level = 0
            
            # Log data
            self.log_data(ear, left_ear, right_ear, head_tilt, fps)
        
        # Draw enhanced status
        self.draw_enhanced_status(frame, ear, left_ear, right_ear, head_tilt, fps)
        
        return frame
    
    def save_session_summary(self):
        """Save session summary to JSON"""
        duration = (datetime.now() - self.stats['session_start']).total_seconds()
        
        summary = {
            'session_start': self.stats['session_start'].isoformat(),
            'session_duration_seconds': duration,
            'driver_name': self.driver_name,
            'total_frames': self.stats['total_frames'],
            'total_blinks': self.stats['total_blinks'],
            'warning_events': self.stats['warning_events'],
            'drowsy_events': self.stats['drowsy_events'],
            'critical_events': self.stats['critical_events'],
            'avg_ear': self.stats['avg_ear'],
            'min_ear': self.stats['min_ear'],
            'max_ear': self.stats['max_ear'],
            'avg_fps': np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        }
        
        filename = f'session_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Session summary saved: {filename}")
        return summary
    
    def run(self):
        """Main enhanced detection loop"""
        logger.info("=" * 60)
        logger.info("Enhanced Driver Drowsiness Detection System v2.0")
        logger.info("=" * 60)
        logger.info(f"Driver: {self.driver_name}")
        logger.info(f"Arduino: {'Enabled' if self.use_arduino else 'Disabled'}")
        logger.info(f"Logging: {'Enabled' if self.enable_logging else 'Disabled'}")
        logger.info("Controls: Q=Quit, S=Screenshot, R=Reset Stats")
        logger.info("=" * 60)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow('Enhanced Drowsiness Detection - Q to quit', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.stats['warning_events'] = 0
                    self.stats['drowsy_events'] = 0
                    self.stats['critical_events'] = 0
                    logger.info("Statistics reset")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            logger.info("Shutting down...")
            
            # Save session summary
            summary = self.save_session_summary()
            logger.info("Session Summary:")
            logger.info(f"  Duration: {summary['session_duration_seconds']:.1f}s")
            logger.info(f"  Total Blinks: {summary['total_blinks']}")
            logger.info(f"  Warnings: {summary['warning_events']}")
            logger.info(f"  Drowsy Events: {summary['drowsy_events']}")
            logger.info(f"  Critical Events: {summary['critical_events']}")
            logger.info(f"  Average EAR: {summary['avg_ear']:.3f}")
            logger.info(f"  Average FPS: {summary['avg_fps']:.1f}")
            
            if self.use_arduino and self.arduino:
                self.send_arduino_command('ALERT_OFF')
                self.arduino.close()
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Shutdown complete")


def main():
    """Main entry point for enhanced version"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Driver Drowsiness Detection System v2.0'
    )
    parser.add_argument('--port', type=str, default='COM8',
                       help='Arduino serial port')
    parser.add_argument('--name', type=str, default='Yusra',
                       help='Driver name')
    parser.add_argument('--no-arduino', action='store_true',
                       help='Run without Arduino')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable data logging')
    
    args = parser.parse_args()
    
    detector = EnhancedDrowsinessDetector(
        arduino_port=args.port,
        driver_name=args.name,
        use_arduino=not args.no_arduino,
        enable_logging=not args.no_logging
    )
    
    detector.run()


if __name__ == '__main__':
    main()
