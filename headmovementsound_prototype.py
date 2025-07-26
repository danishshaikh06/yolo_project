import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import pyttsx3
import threading
from queue import Queue

class SimpleHeadPoseDetector:
    def __init__(self):
        """Initialize the simplified head pose detector with voice feedback"""
        print("🚀 Initializing Simple Head Pose Detector with Voice...")
        
        # Initialize YOLO for face detection
        self.yolo = YOLO('yolov8n.pt')  # Will download automatically
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        
        # Voice feedback system
        self.voice_queue = Queue()
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        
        # Track last spoken direction to avoid repetition
        self.last_spoken_direction = ""
        self.last_speak_time = 0
        self.speak_cooldown = 2.0  # Seconds between voice announcements
        
        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        # Landmark indices for pose estimation
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
        
        # Smoothing
        self.angle_history = {
            'pitch': deque(maxlen=5),
            'yaw': deque(maxlen=5)
        }
        
        # Detection thresholds - only reduced yaw threshold for better left/right
        self.pitch_threshold = 15.0
        self.yaw_threshold = 12.0  # Reduced only for left/right sensitivity
        
        # Camera matrix (will be set when first frame is processed)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        print("✅ Initialization complete with voice feedback!")
    
    def _voice_worker(self):
        """Worker thread for handling voice announcements"""
        while True:
            try:
                message = self.voice_queue.get(timeout=1)
                if message is None:  # Shutdown signal
                    break
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
                self.voice_queue.task_done()
            except:
                continue
    
    def speak_direction(self, direction):
        """Add voice announcement to queue if conditions are met"""
        current_time = time.time()
        
        # Check if we should speak (avoid spam)
        if (direction != self.last_spoken_direction and 
            direction != "Looking at Screen" and 
            current_time - self.last_speak_time >= self.speak_cooldown):
            
            # Add to voice queue
            self.voice_queue.put(direction)
            self.last_spoken_direction = direction
            self.last_speak_time = current_time
            
        # Reset last spoken direction if back to center
        elif direction == "Looking at Screen":
            self.last_spoken_direction = ""
    
    def setup_camera_matrix(self, width, height):
        """Setup camera calibration matrix"""
        focal_length = width
        center = (width / 2, height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def detect_face(self, frame):
        """Detect face using YOLOv8"""
        results = self.yolo(frame, verbose=False)
        
        # Find person class (0) with highest confidence
        best_box = None
        best_conf = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0)
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > best_conf:
                        best_conf = float(box.conf[0])
                        best_box = box.xyxy[0].cpu().numpy()
        
        return best_box, best_conf
    
    def extract_landmarks(self, frame, face_box=None):
        """Extract facial landmarks"""
        # Crop face region if box is provided
        if face_box is not None:
            x1, y1, x2, y2 = map(int, face_box)
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            face_region = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            face_region = frame
            offset_x, offset_y = 0, 0
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Extract specific landmarks
        landmarks = results.multi_face_landmarks[0]
        h, w = face_region.shape[:2]
        
        image_points = []
        for idx in self.landmark_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = int(lm.x * w) + offset_x
                y = int(lm.y * h) + offset_y
                image_points.append([x, y])
        
        if len(image_points) == 6:
            return np.array(image_points, dtype=np.float64)
        return None
    
    def calculate_angles(self, landmarks):
        """Calculate head pose angles with improved yaw calculation"""
        if landmarks is None:
            return None, None, None
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            landmarks,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        if not success:
            return None, None, None
        
        # Convert to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Improved angle calculation
        # Pitch (up/down) - keep original
        pitch = np.degrees(np.arctan2(rmat[2][1], rmat[2][2]))
        
        # Yaw (left/right) - improved calculation
        # Use atan2 for better angle range and accuracy
        yaw = np.degrees(np.arctan2(-rmat[0][2], rmat[0][0]))
        
        # Alternative yaw calculation if the above doesn't work well
        # yaw = np.degrees(np.arcsin(np.clip(rmat[2][0], -1.0, 1.0)))

        roll = np.degrees(np.arctan2(rmat[1][0], rmat[0][0]))
        
        return pitch, yaw, roll
    
    def smooth_angles(self, pitch, yaw):
        """Apply smoothing to angles"""
        if pitch is not None:
            self.angle_history['pitch'].append(pitch)
            pitch = np.mean(list(self.angle_history['pitch']))
        
        if yaw is not None:
            self.angle_history['yaw'].append(yaw)
            yaw = np.mean(list(self.angle_history['yaw']))
        
        return pitch, yaw
    
    def get_direction(self, pitch, yaw, roll, calibrated_angles=(0.0, 0.0, 0.0)):
        """Determine head direction using sequential threshold logic with calibration"""
        if pitch is None or yaw is None or roll is None:
            return "No Face", (0, 0, 255)  # Red
        
        # Use calibrated angles for head pose detection
        pitch_offset, yaw_offset, roll_offset = calibrated_angles
        PITCH_THRESHOLD = 8  # Reduced sensitivity
        YAW_THRESHOLD = 12
        ROLL_THRESHOLD = 5
        
        direction = []

        # Check vertical movement (pitch)
        if pitch > pitch_offset + 10:
            direction.append("Looking Up")
        elif pitch < pitch_offset - 10:
            direction.append("Looking Down")

        # Check horizontal movement (yaw)
        if yaw < yaw_offset - 15:
            direction.append("Looking Left")
        elif yaw > yaw_offset + 15:
            direction.append("Looking Right")

        # Optional: Check roll (tilt)
        #if roll > roll_offset + 10:
         #direction.append("Tilted Left")
        #elif roll < roll_offset - 10:
         # direction.append("Tilted Right")

        # Default state
        if not direction:
            current_state = "Looking at Screen"
            color = (255, 255, 0) 
        else:
            current_state = " + ".join(direction)
            color = (0, 255, 0)  # Green

        return current_state, color
    
    def process_frame(self, frame):
        """Process a single frame"""
        h, w = frame.shape[:2]
        
        # Setup camera matrix if needed
        if self.camera_matrix is None:
            self.setup_camera_matrix(w, h)
        
        # Detect face with YOLO
        face_box, face_conf = self.detect_face(frame)
        
        # Extract landmarks
        landmarks = self.extract_landmarks(frame, face_box)
        
        # Calculate angles
        pitch, yaw, roll = self.calculate_angles(landmarks)
        
        # Smooth angles
        pitch, yaw = self.smooth_angles(pitch, yaw)
        
        # Get direction
        direction, color = self.get_direction(pitch, yaw, roll)
        
        # Add voice feedback
        self.speak_direction(direction)
        
        return {
            'face_box': face_box,
            'face_conf': face_conf,
            'landmarks': landmarks,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'direction': direction,
            'color': color
        }
    
    def draw_results(self, frame, results):
        """Draw results on frame"""
        # Draw face bounding box
        if results['face_box'] is not None:
            x1, y1, x2, y2 = map(int, results['face_box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Face: {results['face_conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw landmarks
        if results['landmarks'] is not None:
            for point in results['landmarks']:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)
        
        # Draw direction
        direction = results['direction']
        color = results['color']
        cv2.putText(frame, direction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Draw angles
        if results['pitch'] is not None and results['yaw'] is not None:
            angle_text = f"Pitch: {results['pitch']:.1f}°  Yaw: {results['yaw']:.1f}°"
            cv2.putText(frame, angle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add voice status indicator
        voice_status = "🔊 Voice: ON" if not self.voice_queue.empty() else "🔊 Voice: Ready"
        cv2.putText(frame, voice_status, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        # Stop voice thread
        self.voice_queue.put(None)
        if self.voice_thread.is_alive():
            self.voice_thread.join(timeout=1)
        
        # Stop TTS engine
        try:
            self.tts_engine.stop()
        except:
            pass

def main():
    """Main function"""
    print("🚀 Starting Simple Head Pose Detection System with Voice")
    
    # Initialize detector
    detector = SimpleHeadPoseDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("📹 Camera opened successfully")
    print("🔊 Voice feedback enabled!")
    print("Controls: Press 'q' to quit")
    print("=" * 50)
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Draw results
            frame = detector.draw_results(frame, results)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                print(f"📊 FPS: {fps:.1f}, Direction: {results['direction']}")
            
            # Display
            cv2.imshow('Head Pose Detection with Voice', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()