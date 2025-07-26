

import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame  # For audio alerts

# Local YOLOv8 - Best performance
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Install: pip install ultralytics pygame")

@dataclass
class DetectionZone:
    name: str
    points: List[Tuple[int, int]]  # Polygon points
    color: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    alert_enabled: bool = True

@dataclass 
class Detection:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float

class AlertManager:
    """Minimal alert system with audio and visual notifications"""
    def __init__(self):
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # 3 seconds between alerts
        self.audio_enabled = False
        
        # Try to initialize pygame audio
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self._create_alert_sound()
            self.audio_enabled = True
            print("✅ Audio alerts enabled")
        except Exception as e:
            print(f"⚠️ Audio disabled (error: {e})")
            print("🔔 Using visual alerts only")
    
    def _create_alert_sound(self):
        """Create a simple beep sound"""
        sample_rate = 22050
        duration = 0.8
        frequency = 1000
        
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2))  # Create stereo array
        
        for i in range(frames):
            # Create a beep pattern (on-off-on)
            envelope = 1.0
            if i < frames * 0.3 or i > frames * 0.7:
                wave_value = envelope * np.sin(2 * np.pi * frequency * i / sample_rate)
            else:
                wave_value = 0.5 * envelope * np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr[i][0] = wave_value  # Left channel
            arr[i][1] = wave_value  # Right channel
        
        arr = (arr * 32767).astype(np.int16)
        self.alert_sound = pygame.sndarray.make_sound(arr)
    
    def _play_system_beep(self):
        """Fallback system beep for Windows"""
        try:
            import winsound
            # Play system beep
            winsound.Beep(800, 300)  # 1000Hz for 500ms
        except:
            # Print visual bell if no audio available
            print("\a🔔 BEEP!")  # Terminal bell character
    
    def trigger_alert(self, zone_name: str, detection_count: int):
        """Trigger alert with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            
            # Try pygame sound first
            if self.audio_enabled:
                try:
                    self.alert_sound.play()
                except:
                    # Fallback to system beep
                    threading.Thread(target=self._play_system_beep, daemon=True).start()
            else:
                # Use system beep as fallback
                threading.Thread(target=self._play_system_beep, daemon=True).start()
            
            # Console alert with more visual emphasis
            alert_msg = f"🚨 ALERT: {detection_count} phone(s) detected in {zone_name} at {time.strftime('%H:%M:%S')}"
            print("=" * 60)
            print(alert_msg)
            print("=" * 60)
            
            self.last_alert_time = current_time

class ZoneManager:
    """Manage detection zones with minimal overhead"""
    def __init__(self):
        self.zones = []
        self.setup_mode = False
        self.current_zone_points = []
    
    def add_zone(self, zone: DetectionZone):
        self.zones.append(zone)
    
    def point_in_zone(self, point: Tuple[int, int], zone: DetectionZone) -> bool:
        """Fast point-in-polygon check"""
        x, y = point
        n = len(zone.points)
        inside = False
        
        p1x, p1y = zone.points[0]
        for i in range(1, n + 1):
            p2x, p2y = zone.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_detections_in_zones(self, detections: List[Detection]) -> dict:
        """Get detections grouped by zone"""
        zone_detections = {}
        
        for detection in detections:
            for zone in self.zones:
                if self.point_in_zone(detection.center, zone):
                    if zone.name not in zone_detections:
                        zone_detections[zone.name] = []
                    zone_detections[zone.name].append(detection)
        
        return zone_detections
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame"""
        overlay = frame.copy()
        
        for zone in self.zones:
            if len(zone.points) >= 3:
                # Draw filled polygon
                pts = np.array(zone.points, np.int32)
                cv2.fillPoly(overlay, [pts], zone.color)
                
                # Draw border
                cv2.polylines(frame, [pts], True, zone.color, 2)
                
                # Add zone label
                center_x = int(np.mean([p[0] for p in zone.points]))
                center_y = int(np.mean([p[1] for p in zone.points]))
                cv2.putText(frame, zone.name, (center_x - 30, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend with original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame

class FastPhoneDetector:
    """Optimized YOLOv8 detector with minimal overhead"""
    def __init__(self, confidence_threshold=0.6):
        if not YOLO_AVAILABLE:
            raise ImportError("Install: pip install ultralytics")
        
        # Use YOLOv8s model as preferred
        self.model = YOLO('yolov8s.pt')
        self.confidence_threshold = confidence_threshold
        self.phone_class_ids = [67]  # Cell phone in COCO
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Ultra-fast detection with minimal processing"""
        # Run inference
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confidences, classes):
                if int(cls) in self.phone_class_ids and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append(Detection(
                        center=(center_x, center_y),
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf)
                    ))
        
        # Update FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        
        return detections

class MinimalPhoneDetectionSystem:
    """Minimal system with maximum performance"""
    def __init__(self, camera_id=0):
        self.detector = FastPhoneDetector()
        self.zone_manager = ZoneManager()
        self.alert_manager = AlertManager()
        
        # Setup camera with optimal settings
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.setup_mode = False
        self.current_zone_points = []
        self.creating_zone = False
        
        # Setup default zones (can be modified)
        self._setup_default_zones()
    
    def _setup_default_zones(self):
        """Create default detection zones"""
        # Example: Desk area zone
        desk_zone = DetectionZone(
            name="Desk Area",
            points=[(100, 200), (540, 200), (540, 400), (100, 400)],
            color=(0, 255, 255),  # Cyan
            alert_enabled=True
        )
        
        # Example: Exam area zone  
        exam_zone = DetectionZone(
            name="Exam Zone",
            points=[(200, 100), (440, 100), (440, 300), (200, 300)],
            color=(255, 0, 255),  # Magenta
            alert_enabled=True
        )
        
        self.zone_manager.add_zone(desk_zone)
        self.zone_manager.add_zone(exam_zone)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for zone creation"""
        if not self.setup_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone_points.append((x, y))
            print(f"Added point: ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.current_zone_points) >= 3:
            # Finish zone creation
            zone_name = f"Zone_{len(self.zone_manager.zones) + 1}"
            new_zone = DetectionZone(
                name=zone_name,
                points=self.current_zone_points.copy(),
                color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            )
            self.zone_manager.add_zone(new_zone)
            print(f"Created zone: {zone_name}")
            self.current_zone_points = []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes with minimal overhead"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            conf = detection.confidence
            
            # Simple green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center point
            cv2.circle(frame, detection.center, 3, (0, 0, 255), -1)
        
        return frame
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw minimal UI information"""
        # FPS
        cv2.putText(frame, f"FPS: {self.detector.current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mode indicator
        mode_text = "SETUP MODE" if self.setup_mode else "DETECTION MODE"
        mode_color = (0, 255, 255) if self.setup_mode else (0, 255, 0)
        cv2.putText(frame, mode_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Instructions
        if self.setup_mode:
            cv2.putText(frame, "Left click: Add point | Right click: Finish zone", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw current zone points
            for i, point in enumerate(self.current_zone_points):
                cv2.circle(frame, point, 5, (0, 255, 255), -1)
                cv2.putText(frame, str(i+1), (point[0]+10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Press 'z' for zone setup | 'q' to quit", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop - optimized for speed"""
        print("🚀 Starting Minimal Phone Detection System")
        print("Controls: 'z' = Zone setup, 'q' = Quit")
        
        cv2.namedWindow("Phone Detection", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Phone Detection", self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Core detection (minimal processing)
            detections = self.detector.detect(frame)
            
            # Zone processing and alerts
            if not self.setup_mode and detections:
                zone_detections = self.zone_manager.get_detections_in_zones(detections)
                
                # Trigger alerts for zones with detections
                for zone_name, zone_dets in zone_detections.items():
                    if zone_dets:
                        # Find the zone object to check if alerts are enabled
                        zone = next((z for z in self.zone_manager.zones if z.name == zone_name), None)
                        if zone and zone.alert_enabled:
                            self.alert_manager.trigger_alert(zone_name, len(zone_dets))
            
            # Draw everything
            frame = self.zone_manager.draw_zones(frame)
            frame = self.draw_detections(frame, detections)
            frame = self.draw_ui(frame)
            
            cv2.imshow("Phone Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.setup_mode = not self.setup_mode
                if self.setup_mode:
                    print("Entered zone setup mode")
                else:
                    print("Exited zone setup mode")
                    self.current_zone_points = []
        
        self.cap.release()
        cv2.destroyAllWindows()

# Quick start function
def main():
    """One-line startup"""
    system = MinimalPhoneDetectionSystem()
    system.run()

if __name__ == "__main__":
    main()