import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, Optional, List
import time
import math
from collections import deque
import statistics

@dataclass
class PersonState:
    # Multiple baseline samples for stability
    baseline_samples: Dict[str, List[float]] = None
    baseline_stats: Dict[str, Dict[str, float]] = None
    
    # Current measurements with confidence
    current_angles: Dict[str, float] = None
    angle_confidence: Dict[str, float] = None
    
    # Advanced tracking for stability
    angle_history: Dict[str, deque] = None
    stability_tracker: Dict[str, int] = None
    
    # Alert system with persistence requirements
    potential_alerts: Dict[str, List[float]] = None  # Store timestamps of potential alerts
    confirmed_alerts: Dict[str, float] = None       # Confirmed alert start times
    alert_counts: Dict[str, int] = None
    
    # Calibration state
    baseline_frames: int = 0
    pose_visible: bool = False
    calibration_complete: bool = False
    
    def __post_init__(self):
        if self.baseline_samples is None:
            self.baseline_samples = {
                'shoulder_tilt': [],
                'head_turn': [],
                'forward_lean': [],
                'neck_angle': []
            }
        if self.baseline_stats is None:
            self.baseline_stats = {}
        if self.current_angles is None:
            self.current_angles = {}
        if self.angle_confidence is None:
            self.angle_confidence = {}
        if self.angle_history is None:
            self.angle_history = {
                'shoulder_tilt': deque(maxlen=15),
                'head_turn': deque(maxlen=15),
                'forward_lean': deque(maxlen=15),
                'neck_angle': deque(maxlen=15)
            }
        if self.stability_tracker is None:
            self.stability_tracker = {}
        if self.potential_alerts is None:
            self.potential_alerts = {
                'LEANING_LEFT': [],
                'LEANING_RIGHT': [],
                'HEAD_TURNING': [],
                'LOOKING_DOWN': [],
                'SUSPICIOUS_FORWARD_LEAN': []
            }
        if self.confirmed_alerts is None:
            self.confirmed_alerts = {}
        if self.alert_counts is None:
            self.alert_counts = {}

class AccurateExamMonitor:
    def __init__(self):
        # More realistic and less sensitive thresholds
        self.SHOULDER_LEAN_THRESHOLD = 15.0     # degrees - significant lean
        self.HEAD_TURN_THRESHOLD = 25.0         # degrees - obvious head turn
        self.FORWARD_LEAN_THRESHOLD = 20.0      # degrees - suspicious forward lean
        self.NECK_DOWN_THRESHOLD = 30.0         # degrees - looking down significantly
        
        # Stability requirements - behavior must be consistent
        self.STABILITY_FRAMES = 8               # frames of consistent behavior
        self.PERSISTENCE_TIME = 2.0             # seconds behavior must persist
        self.CONFIRMATION_SAMPLES = 5           # samples needed to confirm alert
        
        # Calibration - longer for better baseline
        self.BASELINE_FRAMES = 90               # More frames for stable baseline
        self.MIN_CONFIDENCE = 0.7               # Higher confidence requirement
        
        # Initialize MediaPipe with balanced settings
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,                 # Balanced accuracy/speed
            enable_segmentation=False,
            min_detection_confidence=0.8,      # High confidence
            min_tracking_confidence=0.7        # Good tracking stability
        )
        
        # Person state
        self.person = PersonState()
        self.frame_count = 0
        
        # Performance tracking
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Debug mode
        self.debug_mode = True
        
    def calculate_landmark_confidence(self, landmarks) -> float:
        """Calculate overall confidence of pose detection"""
        if not landmarks:
            return 0.0
        
        key_points = [0, 7, 8, 11, 12, 23, 24]  # nose, ears, shoulders, hips
        confidences = []
        
        for idx in key_points:
            if idx < len(landmarks.landmark):
                confidences.append(landmarks.landmark[idx].visibility)
        
        return statistics.mean(confidences) if confidences else 0.0
    
    def get_stable_angle(self, angle_type: str, current_angle: float) -> tuple:
        """Get stable angle with confidence measure"""
        history = self.person.angle_history[angle_type]
        history.append(current_angle)
        
        if len(history) < 5:
            return current_angle, 0.3  # Low confidence for few samples
        
        # Calculate stability - how consistent are recent measurements?
        recent_values = list(history)[-8:]  # Last 8 frames
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        mean_val = statistics.mean(recent_values)
        
        # Confidence based on stability (lower std_dev = higher confidence)
        max_std = 10.0  # Maximum expected standard deviation
        confidence = max(0.0, 1.0 - (std_dev / max_std))
        
        # Use median for more stable measurement
        stable_angle = statistics.median(recent_values)
        
        return stable_angle, confidence
    
    def calculate_precise_angles(self, landmarks, frame_shape) -> Dict[str, tuple]:
        """Calculate angles with confidence scores"""
        h, w = frame_shape[:2]
        angle_data = {}
        
        # Convert to pixel coordinates with visibility check
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            if landmark.visibility > 0.6:  # Higher visibility threshold
                points[idx] = (landmark.x * w, landmark.y * h, landmark.visibility)
        
        try:
            # 1. SHOULDER TILT - Most reliable for side lean detection
            if 11 in points and 12 in points:
                left_shoulder = np.array(points[11][:2])
                right_shoulder = np.array(points[12][:2])
                shoulder_confidence = min(points[11][2], points[12][2])
                
                if shoulder_confidence > 0.7:
                    shoulder_vector = right_shoulder - left_shoulder
                    angle_rad = np.arctan2(shoulder_vector[1], shoulder_vector[0])
                    shoulder_angle = np.degrees(angle_rad)
                    
                    # Normalize to [-45, 45] range for easier interpretation
                    while shoulder_angle > 45:
                        shoulder_angle -= 90
                    while shoulder_angle < -45:
                        shoulder_angle += 90
                    
                    stable_angle, stability = self.get_stable_angle('shoulder_tilt', shoulder_angle)
                    final_confidence = shoulder_confidence * stability
                    angle_data['shoulder_tilt'] = (stable_angle, final_confidence)
            
            # 2. HEAD TURN - For detecting looking left/right
            if 0 in points and 7 in points and 8 in points:
                nose = np.array(points[0][:2])
                left_ear = np.array(points[7][:2])
                right_ear = np.array(points[8][:2])
                head_confidence = min(points[0][2], points[7][2], points[8][2])
                
                if head_confidence > 0.6:
                    # Calculate face direction
                    ear_midpoint = (left_ear + right_ear) / 2
                    face_vector = nose - ear_midpoint
                    
                    # Head turn angle (0 = facing forward)
                    head_angle = np.degrees(np.arctan2(face_vector[0], -face_vector[1]))
                    
                    stable_angle, stability = self.get_stable_angle('head_turn', head_angle)
                    final_confidence = head_confidence * stability
                    angle_data['head_turn'] = (stable_angle, final_confidence)
            
            # 3. FORWARD LEAN - For detecting leaning toward desk/screen
            if 11 in points and 12 in points and 23 in points and 24 in points:
                shoulder_center = (np.array(points[11][:2]) + np.array(points[12][:2])) / 2
                hip_center = (np.array(points[23][:2]) + np.array(points[24][:2])) / 2
                torso_confidence = min(points[11][2], points[12][2], points[23][2], points[24][2])
                
                if torso_confidence > 0.6:
                    torso_vector = shoulder_center - hip_center
                    # Angle from vertical (positive = leaning forward)
                    lean_angle = np.degrees(np.arctan2(abs(torso_vector[0]), torso_vector[1]))
                    
                    stable_angle, stability = self.get_stable_angle('forward_lean', lean_angle)
                    final_confidence = torso_confidence * stability
                    angle_data['forward_lean'] = (stable_angle, final_confidence)
            
            # 4. NECK ANGLE - For detecting looking down (phone/paper)
            if 0 in points and 11 in points and 12 in points:
                nose = np.array(points[0][:2])
                shoulder_center = (np.array(points[11][:2]) + np.array(points[12][:2])) / 2
                neck_confidence = min(points[0][2], points[11][2], points[12][2])
                
                if neck_confidence > 0.6:
                    neck_vector = nose - shoulder_center
                    # Angle from horizontal (positive = looking down)
                    neck_angle = np.degrees(np.arctan2(-neck_vector[1], abs(neck_vector[0])))
                    
                    stable_angle, stability = self.get_stable_angle('neck_angle', neck_angle)
                    final_confidence = neck_confidence * stability
                    angle_data['neck_angle'] = (stable_angle, final_confidence)
        
        except (ValueError, ZeroDivisionError, KeyError):
            pass
        
        return angle_data
    
    def establish_robust_baseline(self, angle_data: Dict[str, tuple]) -> bool:
        """Establish a robust baseline with statistical analysis"""
        if self.person.baseline_frames >= self.BASELINE_FRAMES:
            if not self.person.calibration_complete:
                self.finalize_baseline()
            return True
        
        # Collect baseline samples with high confidence only
        for angle_type, (angle, confidence) in angle_data.items():
            if confidence > 0.8:  # Only use high-confidence measurements
                self.person.baseline_samples[angle_type].append(angle)
        
        self.person.baseline_frames += 1
        return False
    
    def finalize_baseline(self):
        """Calculate baseline statistics from collected samples"""
        for angle_type, samples in self.person.baseline_samples.items():
            if len(samples) >= 20:  # Need enough samples
                # Remove outliers (beyond 2 standard deviations)
                mean_val = statistics.mean(samples)
                std_val = statistics.stdev(samples) if len(samples) > 1 else 0
                
                filtered_samples = [s for s in samples if abs(s - mean_val) <= 2 * std_val]
                
                if filtered_samples:
                    self.person.baseline_stats[angle_type] = {
                        'mean': statistics.mean(filtered_samples),
                        'std': statistics.stdev(filtered_samples) if len(filtered_samples) > 1 else 0,
                        'samples': len(filtered_samples)
                    }
        
        self.person.calibration_complete = True
        print("✅ Baseline calibration completed with statistical analysis")
    
    def detect_suspicious_behavior_advanced(self, angle_data: Dict[str, tuple]) -> Dict[str, float]:
        """Advanced suspicious behavior detection with statistical thresholds"""
        if not self.person.calibration_complete:
            return {}
        
        current_time = time.time()
        suspicious_behaviors = {}
        
        # Update current measurements
        for angle_type, (angle, confidence) in angle_data.items():
            if confidence > self.MIN_CONFIDENCE:
                self.person.current_angles[angle_type] = angle
                self.person.angle_confidence[angle_type] = confidence
        
        # 1. SHOULDER LEAN DETECTION (Left/Right)
        if 'shoulder_tilt' in self.person.current_angles and 'shoulder_tilt' in self.person.baseline_stats:
            current_shoulder = self.person.current_angles['shoulder_tilt']
            baseline = self.person.baseline_stats['shoulder_tilt']
            confidence = self.person.angle_confidence['shoulder_tilt']
            
            # Use statistical threshold (mean ± 2*std + fixed threshold)
            deviation = abs(current_shoulder - baseline['mean'])
            threshold = max(self.SHOULDER_LEAN_THRESHOLD, baseline['std'] * 2.5)
            
            if deviation > threshold and confidence > 0.8:
                intensity = min(1.0, deviation / (threshold * 1.5))
                if current_shoulder > baseline['mean']:
                    suspicious_behaviors['LEANING_RIGHT'] = intensity
                else:
                    suspicious_behaviors['LEANING_LEFT'] = intensity
        
        # 2. HEAD TURNING DETECTION
        if 'head_turn' in self.person.current_angles and 'head_turn' in self.person.baseline_stats:
            current_head = self.person.current_angles['head_turn']
            baseline = self.person.baseline_stats['head_turn']
            confidence = self.person.angle_confidence['head_turn']
            
            deviation = abs(current_head - baseline['mean'])
            threshold = max(self.HEAD_TURN_THRESHOLD, baseline['std'] * 3.0)
            
            if deviation > threshold and confidence > 0.7:
                intensity = min(1.0, deviation / (threshold * 1.2))
                suspicious_behaviors['HEAD_TURNING'] = intensity
        
        # 3. FORWARD LEAN DETECTION (Suspicious leaning toward desk)
        if 'forward_lean' in self.person.current_angles and 'forward_lean' in self.person.baseline_stats:
            current_lean = self.person.current_angles['forward_lean']
            baseline = self.person.baseline_stats['forward_lean']
            confidence = self.person.angle_confidence['forward_lean']
            
            # Only trigger if significantly more forward than baseline
            forward_increase = current_lean - baseline['mean']
            threshold = max(self.FORWARD_LEAN_THRESHOLD, baseline['std'] * 2.0)
            
            if forward_increase > threshold and confidence > 0.7:
                intensity = min(1.0, forward_increase / (threshold * 1.3))
                suspicious_behaviors['SUSPICIOUS_FORWARD_LEAN'] = intensity
        
        # 4. LOOKING DOWN DETECTION (Phone/paper checking)
        if 'neck_angle' in self.person.current_angles and 'neck_angle' in self.person.baseline_stats:
            current_neck = self.person.current_angles['neck_angle']
            baseline = self.person.baseline_stats['neck_angle']
            confidence = self.person.angle_confidence['neck_angle']
            
            # Looking down more than baseline
            down_increase = current_neck - baseline['mean']
            threshold = max(self.NECK_DOWN_THRESHOLD, baseline['std'] * 2.5)
            
            if down_increase > threshold and confidence > 0.7:
                intensity = min(1.0, down_increase / (threshold * 1.2))
                suspicious_behaviors['LOOKING_DOWN'] = intensity
        
        return suspicious_behaviors
    
    def update_advanced_alerts(self, behaviors: Dict[str, float]) -> set:
        """Advanced alert system requiring persistence and consistency"""
        current_time = time.time()
        confirmed_alerts = set()
        
        # Clean old potential alerts (older than 5 seconds)
        for behavior_type in self.person.potential_alerts:
            self.person.potential_alerts[behavior_type] = [
                t for t in self.person.potential_alerts[behavior_type] 
                if current_time - t < 5.0
            ]
        
        # Process each detected behavior
        for behavior, intensity in behaviors.items():
            if intensity > 0.6:  # Minimum intensity threshold
                # Add to potential alerts
                self.person.potential_alerts[behavior].append(current_time)
                
                # Check if we have enough consistent detections
                recent_alerts = [
                    t for t in self.person.potential_alerts[behavior]
                    if current_time - t <= self.PERSISTENCE_TIME
                ]
                
                if len(recent_alerts) >= self.CONFIRMATION_SAMPLES:
                    # Confirm the alert
                    if behavior not in self.person.confirmed_alerts:
                        self.person.confirmed_alerts[behavior] = current_time
                        self.person.alert_counts[behavior] = self.person.alert_counts.get(behavior, 0) + 1
                        print(f"🚨 CONFIRMED ALERT: {behavior} (intensity: {intensity:.2f})")
                    
                    confirmed_alerts.add(behavior)
        
        # Remove confirmed alerts that are no longer detected
        expired = []
        for behavior in self.person.confirmed_alerts:
            if behavior not in behaviors or behaviors[behavior] < 0.4:
                # Check if behavior has been absent for long enough
                recent_detections = [
                    t for t in self.person.potential_alerts.get(behavior, [])
                    if current_time - t <= 1.0
                ]
                if not recent_detections:
                    expired.append(behavior)
        
        for behavior in expired:
            del self.person.confirmed_alerts[behavior]
        
        return confirmed_alerts
    
    def draw_professional_display(self, frame: np.ndarray, landmarks, angle_data: Dict[str, tuple], 
                                behaviors: Dict[str, float], confirmed_alerts: set) -> np.ndarray:
        """Professional monitoring display with clear status information"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw pose landmarks (subtle)
        if landmarks and self.debug_mode:
            self.mp_drawing.draw_landmarks(
                overlay, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=1)
            )
        
        # Main status panel
        panel_width = 450
        panel_height = 280
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay_panel = overlay.copy()
        cv2.rectangle(overlay_panel, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.7, overlay_panel, 0.3, 0)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Header
        cv2.putText(overlay, "EXAM MONITORING SYSTEM", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calibration status
        y_pos = panel_y + 55
        if not self.person.calibration_complete:
            progress = (self.person.baseline_frames / self.BASELINE_FRAMES) * 100
            status_text = f"CALIBRATING: {progress:.1f}%"
            status_color = (0, 255, 255)
            cv2.putText(overlay, "Please sit normally during calibration", 
                       (panel_x + 10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            status_text = "MONITORING ACTIVE"
            status_color = (0, 255, 0) if not confirmed_alerts else (0, 100, 255)
        
        cv2.putText(overlay, status_text, (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Current measurements (only if calibrated)
        if self.person.calibration_complete:
            y_pos += 40
            cv2.putText(overlay, "Current Measurements:", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_pos += 20
            measurements = [
                ("Shoulder", "shoulder_tilt", "°"),
                ("Head Turn", "head_turn", "°"),
                ("Forward Lean", "forward_lean", "°"),
                ("Looking Down", "neck_angle", "°")
            ]
            
            for label, angle_type, unit in measurements:
                if angle_type in self.person.current_angles:
                    current = self.person.current_angles[angle_type]
                    confidence = self.person.angle_confidence.get(angle_type, 0)
                    baseline = self.person.baseline_stats.get(angle_type, {}).get('mean', 0)
                    
                    # Color based on deviation
                    deviation = abs(current - baseline)
                    if deviation > 20:
                        color = (0, 0, 255)  # Red for large deviation
                    elif deviation > 10:
                        color = (0, 255, 255)  # Yellow for moderate deviation
                    else:
                        color = (0, 255, 0)  # Green for normal
                    
                    text = f"{label}: {current:.1f}{unit} (conf: {confidence:.2f})"
                    cv2.putText(overlay, text, (panel_x + 15, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_pos += 18
        
        # Alert section
        if confirmed_alerts:
            alert_y = panel_y + panel_height + 20
            alert_panel_height = len(confirmed_alerts) * 30 + 40
            
            # Alert panel background
            cv2.rectangle(overlay, (panel_x, alert_y), 
                         (panel_x + panel_width, alert_y + alert_panel_height), (0, 0, 50), -1)
            cv2.rectangle(overlay, (panel_x, alert_y), 
                         (panel_x + panel_width, alert_y + alert_panel_height), (0, 0, 255), 2)
            
            cv2.putText(overlay, "🚨 SUSPICIOUS BEHAVIOR DETECTED", (panel_x + 10, alert_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            alert_text_y = alert_y + 45
            for alert in confirmed_alerts:
                intensity = behaviors.get(alert, 0)
                alert_display = alert.replace('_', ' ').title()
                text = f"• {alert_display}: {intensity*100:.0f}% intensity"
                cv2.putText(overlay, text, (panel_x + 15, alert_text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                alert_text_y += 25
        
        # System info (bottom left)
        info_y = h - 80
        cv2.putText(overlay, f"FPS: {self.current_fps:.1f} | Frame: {self.frame_count}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(overlay, "Controls: Q=Quit | R=Reset | D=Debug Toggle", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Overall status indicator (top left)
        if self.person.calibration_complete:
            if confirmed_alerts:
                status_indicator = "⚠️ ALERT"
                indicator_color = (0, 0, 255)
            else:
                status_indicator = "✅ NORMAL"
                indicator_color = (0, 255, 0)
        else:
            status_indicator = "🔄 CALIBRATING"
            indicator_color = (0, 255, 255)
        
        cv2.putText(overlay, status_indicator, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, indicator_color, 3)
        
        return overlay
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing pipeline with enhanced accuracy"""
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        angle_data = {}
        behaviors = {}
        confirmed_alerts = set()
        
        if results.pose_landmarks:
            # Check overall pose quality
            pose_confidence = self.calculate_landmark_confidence(results.pose_landmarks)
            
            if pose_confidence > 0.6:  # Only process high-quality poses
                self.person.pose_visible = True
                
                # Calculate precise angles
                angle_data = self.calculate_precise_angles(results.pose_landmarks, frame.shape)
                
                # Establish or use baseline
                baseline_complete = self.establish_robust_baseline(angle_data)
                
                if baseline_complete:
                    # Detect suspicious behaviors
                    behaviors = self.detect_suspicious_behavior_advanced(angle_data)
                    
                    # Update advanced alert system
                    confirmed_alerts = self.update_advanced_alerts(behaviors)
            else:
                self.person.pose_visible = False
        else:
            self.person.pose_visible = False
        
        # Calculate FPS
        self.fps_counter += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = time.time()
        
        # Draw professional display
        result = self.draw_professional_display(
            frame, 
            results.pose_landmarks if results.pose_landmarks else None, 
            angle_data, 
            behaviors, 
            confirmed_alerts
        )
        
        return result
    
    def reset_system(self):
        """Complete system reset"""
        self.person = PersonState()
        print("🔄 System reset - Starting fresh calibration")
    
    def toggle_debug(self):
        """Toggle debug display mode"""
        self.debug_mode = not self.debug_mode
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def get_detailed_summary(self) -> Dict:
        """Get comprehensive session summary"""
        return {
            'session_stats': {
                'total_frames': self.frame_count,
                'calibration_complete': self.person.calibration_complete,
                'baseline_frames_collected': self.person.baseline_frames
            },
            'baseline_statistics': dict(self.person.baseline_stats),
            'alert_summary': dict(self.person.alert_counts),
            'final_measurements': dict(self.person.current_angles)
        }

def main():
    print("🎯 ACCURATE SINGLE PERSON EXAM MONITOR")
    print("=" * 55)
    print("✨ Enhanced with statistical analysis and persistence checking")
    
    monitor = AccurateExamMonitor()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera initialized")
    print("\n📋 IMPORTANT INSTRUCTIONS:")
    print("   1. Sit in your normal, comfortable exam position")
    print("   2. Stay relatively still for 3 seconds (90 frames) during calibration")
    print("   3. The system learns YOUR normal posture statistically")
    print("   4. Only significant deviations will trigger alerts")
    print("   5. Alerts require consistent behavior over 2+ seconds")
    print("\n🎮 CONTROLS:")
    print("   Q = Quit monitoring")
    print("   R = Reset and recalibrate")
    print("   D = Toggle debug display")
    print("\n🚀 Starting enhanced monitoring...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Warning: Failed to read frame")
                continue
            
            # Process frame with enhanced accuracy
            result = monitor.process_frame(frame)
            
            # Display result
            cv2.imshow('Accurate Exam Monitor - Statistical Analysis', result)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                monitor.reset_system()
            elif key == ord('d'):
                monitor.toggle_debug()
            elif key == 27:  # Escape key
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Detailed session summary
        summary = monitor.get_detailed_summary()
        print("\n" + "=" * 55)
        print("📊 DETAILED SESSION SUMMARY")
        print("=" * 55)
        
        session = summary['session_stats']
        print(f"Total frames processed: {session['total_frames']}")
        print(f"Calibration completed: {session['calibration_complete']}")
        print(f"Baseline frames collected: {session['baseline_frames_collected']}")
        
        if summary['baseline_statistics']:
            print("\n📐 BASELINE STATISTICS:")
            for angle_type, stats in summary['baseline_statistics'].items():
                print(f"  {angle_type}: mean={stats['mean']:.1f}°, std={stats['std']:.1f}°")
        
        if summary['alert_summary']:
            print("\n⚠️ ALERT SUMMARY:")
            for alert_type, count in summary['alert_summary'].items():
                print(f"  {alert_type}: {count} times")
        else:
            print("\n✅ No suspicious behavior detected during session")
        
        print("\n🎓 Session completed successfully!")

if __name__ == "__main__":
    main()