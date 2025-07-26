import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class MinimalExamProctor:
    def __init__(self):
        """Initialize minimal exam proctoring system"""
        print("🚀 Initializing Minimal Exam Proctoring...")
        
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Key body landmarks (MediaPipe indices)
        self.landmarks = {
            'shoulders': [11, 12],  # Left, Right shoulder
            'elbows': [13, 14],     # Left, Right elbow  
            'wrists': [15, 16],     # Left, Right wrist
            'nose': 0,              # Nose tip
            'hips': [23, 24]        # Left, Right hip
        }
        
        # Movement tracking (smaller buffers for responsiveness)
        self.history = {
            'head_pos': deque(maxlen=10),
            'shoulder_angle': deque(maxlen=10),
            'arm_positions': deque(maxlen=10),
            'movement_speed': deque(maxlen=20)
        }
        
        # Baseline (auto-calibrated)
        self.baseline = {
            'head_center': None,
            'shoulder_angle': None,
            'calibrated': False,
            'frames_count': 0
        }
        
        # Simplified thresholds
        self.thresholds = {
            'head_turn': 80,        # pixels from center
            'body_turn': 25,        # degrees
            'reach_distance': 150,   # pixels from shoulder
            'movement_spike': 3.0    # movement multiplier
        }
        
        # Alert counters
        self.alerts = {'turn': 0, 'reach': 0, 'fidget': 0}
        
        print("✅ Ready - Calibrating for 100 frames...")
    
    def get_landmark_pos(self, landmarks, index, width, height):
        """Get landmark position with visibility check"""
        if index >= len(landmarks.landmark):
            return None
        
        lm = landmarks.landmark[index]
        if lm.visibility < 0.7:
            return None
            
        return [int(lm.x * width), int(lm.y * height)]
    
    def calculate_angle(self, p1, p2):
        """Calculate angle between two points"""
        try:
            vector = np.array(p2) - np.array(p1)
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            return angle
        except:
            return None
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance"""
        try:
            return np.linalg.norm(np.array(p1) - np.array(p2))
        except:
            return 0
    
    def auto_calibrate(self, head_pos, shoulder_angle):
        """Auto-calibrate baseline from first 100 frames"""
        if self.baseline['calibrated']:
            return
        
        self.baseline['frames_count'] += 1
        
        if head_pos:
            self.history['head_pos'].append(head_pos)
        if shoulder_angle is not None:
            self.history['shoulder_angle'].append(shoulder_angle)
        
        # Calibrate after 100 frames
        if self.baseline['frames_count'] >= 100:
            if self.history['head_pos'] and self.history['shoulder_angle']:
                # Use median for robust baseline
                head_positions = list(self.history['head_pos'])
                shoulder_angles = list(self.history['shoulder_angle'])
                
                self.baseline['head_center'] = np.median(head_positions, axis=0)
                self.baseline['shoulder_angle'] = np.median(shoulder_angles)
                self.baseline['calibrated'] = True
                
                print("✅ Calibration complete - Monitoring active")
    
    def detect_suspicious_behavior(self, head_pos, shoulder_angle, arm_positions, movement_speed):
        """Detect suspicious behaviors with high accuracy"""
        if not self.baseline['calibrated']:
            return []
        
        alerts = []
        
        # 1. Head/Body turning detection
        if head_pos and self.baseline['head_center'] is not None:
            head_deviation = self.calculate_distance(head_pos, self.baseline['head_center'])
            
            if head_deviation > self.thresholds['head_turn']:
                alerts.append("TURN")
                self.alerts['turn'] += 1
        
        if shoulder_angle is not None:
            angle_diff = abs(shoulder_angle - self.baseline['shoulder_angle'])
            if angle_diff > self.thresholds['body_turn']:
                if "TURN" not in alerts:  # Avoid duplicate alerts
                    alerts.append("TURN")
                    self.alerts['turn'] += 1
        
        # 2. Reaching behavior detection
        if arm_positions:
            for arm_pos in arm_positions:
                if arm_pos['shoulder'] and arm_pos['wrist']:
                    reach_distance = self.calculate_distance(
                        arm_pos['shoulder'], arm_pos['wrist']
                    )
                    if reach_distance > self.thresholds['reach_distance']:
                        alerts.append("REACH")
                        self.alerts['reach'] += 1
                        break
        
        # 3. Excessive movement detection
        if len(self.history['movement_speed']) > 10:
            recent_movement = np.mean(list(self.history['movement_speed'])[-5:])
            baseline_movement = np.mean(list(self.history['movement_speed'])[:-5])
            
            if recent_movement > baseline_movement * self.thresholds['movement_spike']:
                alerts.append("FIDGET")
                self.alerts['fidget'] += 1
        
        return alerts
    
    def process_frame(self, frame):
        """Process single frame - main detection pipeline"""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get pose landmarks
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                'pose_detected': False,
                'alerts': [],
                'calibrated': self.baseline['calibrated']
            }
        
        landmarks = results.pose_landmarks
        
        # Extract key positions
        nose_pos = self.get_landmark_pos(landmarks, self.landmarks['nose'], width, height)
        left_shoulder = self.get_landmark_pos(landmarks, self.landmarks['shoulders'][0], width, height)
        right_shoulder = self.get_landmark_pos(landmarks, self.landmarks['shoulders'][1], width, height)
        left_elbow = self.get_landmark_pos(landmarks, self.landmarks['elbows'][0], width, height)
        right_elbow = self.get_landmark_pos(landmarks, self.landmarks['elbows'][1], width, height)
        left_wrist = self.get_landmark_pos(landmarks, self.landmarks['wrists'][0], width, height)
        right_wrist = self.get_landmark_pos(landmarks, self.landmarks['wrists'][1], width, height)
        
        # Calculate shoulder angle
        shoulder_angle = None
        if left_shoulder and right_shoulder:
            shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder)
        
        # Prepare arm positions
        arm_positions = []
        if left_shoulder and left_wrist:
            arm_positions.append({'shoulder': left_shoulder, 'wrist': left_wrist})
        if right_shoulder and right_wrist:
            arm_positions.append({'shoulder': right_shoulder, 'wrist': right_wrist})
        
        # Calculate movement speed
        movement_speed = 0
        if arm_positions and len(self.history['arm_positions']) > 0:
            prev_arms = list(self.history['arm_positions'])[-1]
            for i, arm in enumerate(arm_positions):
                if i < len(prev_arms) and prev_arms[i]['wrist']:
                    movement_speed += self.calculate_distance(
                        arm['wrist'], prev_arms[i]['wrist']
                    )
        
        # Update history
        if nose_pos:
            self.history['head_pos'].append(nose_pos)
        if shoulder_angle is not None:
            self.history['shoulder_angle'].append(shoulder_angle)
        if arm_positions:
            self.history['arm_positions'].append(arm_positions)
        self.history['movement_speed'].append(movement_speed)
        
        # Auto-calibrate
        self.auto_calibrate(nose_pos, shoulder_angle)
        
        # Detect suspicious behavior
        alerts = self.detect_suspicious_behavior(
            nose_pos, shoulder_angle, arm_positions, movement_speed
        )
        
        return {
            'pose_detected': True,
            'nose_pos': nose_pos,
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_wrist': left_wrist,
            'right_wrist': right_wrist,
            'shoulder_angle': shoulder_angle,
            'movement_speed': movement_speed,
            'alerts': alerts,
            'calibrated': self.baseline['calibrated']
        }
    
    def draw_minimal_overlay(self, frame, results):
        """Draw minimal, clean overlay"""
        if not results['pose_detected']:
            cv2.putText(frame, "NO PERSON DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Draw key landmarks
        if results['nose_pos']:
            cv2.circle(frame, tuple(results['nose_pos']), 4, (255, 255, 0), -1)
        
        if results['left_shoulder'] and results['right_shoulder']:
            cv2.line(frame, tuple(results['left_shoulder']), 
                    tuple(results['right_shoulder']), (0, 255, 0), 2)
        
        # Draw wrists
        for wrist_pos in [results['left_wrist'], results['right_wrist']]:
            if wrist_pos:
                cv2.circle(frame, tuple(wrist_pos), 6, (0, 255, 255), -1)
        
        # Status display
        status_color = (0, 255, 0) if results['calibrated'] else (255, 255, 0)
        status_text = "MONITORING" if results['calibrated'] else "CALIBRATING"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Alert display
        if results['alerts']:
            alert_text = " | ".join(results['alerts'])
            cv2.putText(frame, f"ALERT: {alert_text}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Alert counters (bottom)
        counter_text = f"Turn:{self.alerts['turn']} Reach:{self.alerts['reach']} Fidget:{self.alerts['fidget']}"
        cv2.putText(frame, counter_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Movement indicator
        if results['movement_speed'] > 0:
            movement_bar_length = min(int(results['movement_speed'] * 2), 200)
            cv2.rectangle(frame, (frame.shape[1] - 220, 10), 
                         (frame.shape[1] - 220 + movement_bar_length, 30), 
                         (0, 255, 255), -1)
            cv2.putText(frame, "MOVEMENT", (frame.shape[1] - 220, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def main():
    """Main execution function"""
    print("🎯 Minimal Accurate Exam Proctoring System")
    
    # Initialize system
    proctor = MinimalExamProctor()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    print("📹 Camera ready | Press 'q' to quit | 'r' to reset")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = proctor.process_frame(frame)
            
            # Draw overlay
            frame = proctor.draw_minimal_overlay(frame, results)
            
            # Display
            cv2.imshow('Exam Proctoring', frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset system
                proctor.baseline = {
                    'head_center': None,
                    'shoulder_angle': None,
                    'calibrated': False,
                    'frames_count': 0
                }
                proctor.history = {k: deque(maxlen=v.maxlen) 
                                 for k, v in proctor.history.items()}
                proctor.alerts = {'turn': 0, 'reach': 0, 'fidget': 0}
                print("🔄 System reset - Recalibrating...")
            
            frame_count += 1
            
            # Performance update every 5 seconds
            if frame_count % 150 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                total_alerts = sum(proctor.alerts.values())
                print(f"📊 FPS: {fps:.1f} | Total Alerts: {total_alerts}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print("\n" + "="*40)
        print("📋 SESSION SUMMARY")
        print("="*40)
        for alert_type, count in proctor.alerts.items():
            print(f"{alert_type.upper()}: {count}")
        print("✅ Session complete")

if __name__ == "__main__":
    main()