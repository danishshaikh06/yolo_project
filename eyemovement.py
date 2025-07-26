import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

class EnhancedEyeMovementDetector:
    def __init__(self):
        """Initialize the enhanced eye movement detector"""
        print("🚀 Initializing Enhanced Eye Movement Detector...")
        
        # Initialize YOLO for face detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Face Mesh with better parameters
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.7
        )
        
        # More comprehensive eye landmark indices for MediaPipe
        # Left eye landmarks (more points for better accuracy)
        self.left_eye_indices = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        # Right eye landmarks
        self.right_eye_indices = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        # Iris landmarks for more precise tracking
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        
        # Smoothing buffers for stable detection
        self.gaze_history = deque(maxlen=5)
        self.left_pupil_history = deque(maxlen=3)
        self.right_pupil_history = deque(maxlen=3)
        
        # Calibration parameters
        self.eye_aspect_ratio_threshold = 0.25
        self.blink_counter = 0
        
        print("✅ Enhanced initialization complete!")
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        if len(eye_points) < 6:
            return 0
        
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if C == 0:
            return 0
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_face(self, frame):
        """Enhanced face detection with better filtering"""
        results = self.yolo(frame, verbose=False)
        
        best_box = None
        best_conf = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.7:  # Higher confidence threshold
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            box_coords = box.xyxy[0].cpu().numpy()
                            
                            # Filter out boxes that are too small or have wrong aspect ratio
                            w = box_coords[2] - box_coords[0]
                            h = box_coords[3] - box_coords[1]
                            aspect_ratio = w / h
                            
                            if w > 100 and h > 100 and 0.7 < aspect_ratio < 1.5:
                                best_box = box_coords
        
        return best_box, best_conf
    
    def extract_eye_landmarks(self, frame, face_box=None):
        """Enhanced eye landmark extraction with iris detection"""
        if face_box is not None:
            x1, y1, x2, y2 = map(int, face_box)
            padding = 30  # Increased padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            face_region = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            face_region = frame
            offset_x, offset_y = 0, 0
        
        # Enhanced preprocessing
        face_region = cv2.bilateralFilter(face_region, 5, 80, 80)
        rgb_frame = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None, None, None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = face_region.shape[:2]
        
        # Extract eye contour points
        left_eye_points = []
        for idx in self.left_eye_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = int(lm.x * w) + offset_x
                y = int(lm.y * h) + offset_y
                left_eye_points.append([x, y])
        
        right_eye_points = []
        for idx in self.right_eye_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = int(lm.x * w) + offset_x
                y = int(lm.y * h) + offset_y
                right_eye_points.append([x, y])
        
        # Extract iris points for more accurate pupil tracking
        left_iris_points = []
        for idx in self.left_iris_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = int(lm.x * w) + offset_x
                y = int(lm.y * h) + offset_y
                left_iris_points.append([x, y])
        
        right_iris_points = []
        for idx in self.right_iris_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x = int(lm.x * w) + offset_x
                y = int(lm.y * h) + offset_y
                right_iris_points.append([x, y])
        
        # Convert to numpy arrays
        if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
            return (np.array(left_eye_points), np.array(right_eye_points),
                    np.array(left_iris_points) if left_iris_points else None,
                    np.array(right_iris_points) if right_iris_points else None)
        
        return None, None, None, None
    
    def enhanced_pupil_detection(self, eye_region, iris_points=None):
        """Enhanced pupil detection using multiple methods"""
        if eye_region is None or eye_region.size == 0:
            return None, None
        
        # Method 1: Use iris landmarks if available
        if iris_points is not None and len(iris_points) > 0:
            # Calculate center of iris points
            center = np.mean(iris_points, axis=0).astype(int)
            # Create a bounding box around iris
            x_coords = iris_points[:, 0]
            y_coords = iris_points[:, 1]
            bbox = (min(x_coords), min(y_coords), 
                   max(x_coords) - min(x_coords), 
                   max(y_coords) - min(y_coords))
            return tuple(center), bbox
        
        # Method 2: Enhanced image processing
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Multiple blur and threshold attempts
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50 and area < 1000:  # Reasonable pupil size
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Reasonably circular
                            valid_contours.append((contour, area))
            
            if valid_contours:
                # Get the most circular contour
                best_contour = max(valid_contours, key=lambda x: x[1])[0]
                
                # Get center using moments
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(best_contour)
                    
                    return (center_x, center_y), (x, y, w, h)
        
        return None, None
    
    def smooth_pupil_position(self, pupil_pos, history_buffer):
        """Smooth pupil position using history"""
        if pupil_pos is None:
            return None
        
        history_buffer.append(pupil_pos)
        
        if len(history_buffer) >= 2:
            # Use weighted average with more weight on recent positions
            weights = np.linspace(0.5, 1.0, len(history_buffer))
            positions = np.array(list(history_buffer))
            weighted_pos = np.average(positions, axis=0, weights=weights)
            return tuple(weighted_pos.astype(int))
        
        return pupil_pos
    
    def calculate_gaze_direction_enhanced(self, left_pupil, right_pupil, left_eye_points, right_eye_points):
        """Enhanced gaze direction calculation with better thresholds"""
        if not left_pupil or not right_pupil or left_eye_points is None or right_eye_points is None:
            return "No Gaze Detected"
        
        # Get eye bounding rectangles
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)
        
        # Calculate relative positions within each eye
        left_rel_x = (left_pupil[0] - left_eye_rect[0]) / left_eye_rect[2]
        left_rel_y = (left_pupil[1] - left_eye_rect[1]) / left_eye_rect[3]
        
        right_rel_x = (right_pupil[0] - right_eye_rect[0]) / right_eye_rect[2]
        right_rel_y = (right_pupil[1] - right_eye_rect[1]) / right_eye_rect[3]
        
        # Average the relative positions
        avg_rel_x = (left_rel_x + right_rel_x) / 2
        avg_rel_y = (left_rel_y + right_rel_y) / 2
        
        # More precise thresholds
        horizontal_threshold = 0.15
        vertical_threshold = 0.15
        
        # Determine gaze direction with hysteresis
        if avg_rel_x < 0.5 - horizontal_threshold:
            horizontal_gaze = "Left"
        elif avg_rel_x > 0.5 + horizontal_threshold:
            horizontal_gaze = "Right"
        else:
            horizontal_gaze = "Center"
        
        if avg_rel_y < 0.5 - vertical_threshold:
            vertical_gaze = "Up"
        elif avg_rel_y > 0.5 + vertical_threshold:
            vertical_gaze = "Down"
        else:
            vertical_gaze = "Center"
        
        # Combine directions
        if horizontal_gaze == "Center" and vertical_gaze == "Center":
            gaze_direction = "Looking Center"
        elif horizontal_gaze == "Center":
            gaze_direction = f"Looking {vertical_gaze}"
        elif vertical_gaze == "Center":
            gaze_direction = f"Looking {horizontal_gaze}"
        else:
            gaze_direction = f"Looking {vertical_gaze}-{horizontal_gaze}"
        
        return gaze_direction
    
    def smooth_gaze_direction(self, gaze_direction):
        """Smooth gaze direction using history"""
        self.gaze_history.append(gaze_direction)
        
        if len(self.gaze_history) >= 3:
            # Use majority voting for stability
            from collections import Counter
            gaze_counts = Counter(self.gaze_history)
            most_common_gaze = gaze_counts.most_common(1)[0][0]
            return most_common_gaze
        
        return gaze_direction
    
    def process_frame(self, frame):
        """Process a single frame with enhanced detection"""
        # Detect face
        face_box, face_conf = self.detect_face(frame)
        
        # Extract eye landmarks and iris points
        left_eye_points, right_eye_points, left_iris_points, right_iris_points = \
            self.extract_eye_landmarks(frame, face_box)
        
        gaze_direction = "No Eyes Detected"
        left_pupil = None
        right_pupil = None
        left_ear = 0
        right_ear = 0
        
        if left_eye_points is not None and right_eye_points is not None:
            # Calculate Eye Aspect Ratios for blink detection
            left_ear = self.calculate_eye_aspect_ratio(left_eye_points[:6])
            right_ear = self.calculate_eye_aspect_ratio(right_eye_points[:6])
            
            # Check if eyes are open
            if left_ear > self.eye_aspect_ratio_threshold and right_ear > self.eye_aspect_ratio_threshold:
                # Extract eye regions
                left_eye_rect = cv2.boundingRect(left_eye_points)
                right_eye_rect = cv2.boundingRect(right_eye_points)
                
                left_eye_region = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], 
                                     left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
                right_eye_region = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], 
                                       right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
                
                # Detect pupils with enhanced method
                left_pupil_raw, _ = self.enhanced_pupil_detection(left_eye_region, 
                                                                left_iris_points - [left_eye_rect[0], left_eye_rect[1]] if left_iris_points is not None else None)
                right_pupil_raw, _ = self.enhanced_pupil_detection(right_eye_region,
                                                                 right_iris_points - [right_eye_rect[0], right_eye_rect[1]] if right_iris_points is not None else None)
                
                # Convert to global coordinates and smooth
                if left_pupil_raw:
                    left_pupil_global = (left_pupil_raw[0] + left_eye_rect[0], 
                                       left_pupil_raw[1] + left_eye_rect[1])
                    left_pupil = self.smooth_pupil_position(left_pupil_global, self.left_pupil_history)
                
                if right_pupil_raw:
                    right_pupil_global = (right_pupil_raw[0] + right_eye_rect[0], 
                                        right_pupil_raw[1] + right_eye_rect[1])
                    right_pupil = self.smooth_pupil_position(right_pupil_global, self.right_pupil_history)
                
                # Calculate gaze direction
                if left_pupil and right_pupil:
                    raw_gaze = self.calculate_gaze_direction_enhanced(left_pupil, right_pupil, 
                                                                   left_eye_points, right_eye_points)
                    gaze_direction = self.smooth_gaze_direction(raw_gaze)
                else:
                    gaze_direction = "Pupils Not Detected"
            else:
                gaze_direction = "Eyes Closed"
                self.blink_counter += 1
        
        return {
            'face_box': face_box,
            'face_conf': face_conf,
            'left_eye_points': left_eye_points,
            'right_eye_points': right_eye_points,
            'left_iris_points': left_iris_points,
            'right_iris_points': right_iris_points,
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'gaze_direction': gaze_direction,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'blink_count': self.blink_counter
        }
    
    def draw_results(self, frame, results):
        """Draw enhanced results on frame"""
        # Draw face bounding box
        if results['face_box'] is not None:
            x1, y1, x2, y2 = map(int, results['face_box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Face: {results['face_conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw eye contours
        if results['left_eye_points'] is not None:
            cv2.polylines(frame, [results['left_eye_points']], True, (0, 255, 0), 1)
        
        if results['right_eye_points'] is not None:
            cv2.polylines(frame, [results['right_eye_points']], True, (0, 255, 0), 1)
        
        # Draw iris points
        if results['left_iris_points'] is not None:
            for point in results['left_iris_points']:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
        
        if results['right_iris_points'] is not None:
            for point in results['right_iris_points']:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
        
        # Draw pupils
        if results['left_pupil']:
            cv2.circle(frame, results['left_pupil'], 5, (0, 0, 255), -1)
            cv2.circle(frame, results['left_pupil'], 8, (0, 0, 255), 2)
        
        if results['right_pupil']:
            cv2.circle(frame, results['right_pupil'], 5, (0, 0, 255), -1)
            cv2.circle(frame, results['right_pupil'], 8, (0, 0, 255), 2)
        
        # Draw gaze direction with color coding
        gaze_direction = results['gaze_direction']
        
        if "Left" in gaze_direction:
            color = (0, 255, 255)  # Yellow
        elif "Right" in gaze_direction:
            color = (255, 0, 255)  # Magenta
        elif "Up" in gaze_direction:
            color = (255, 255, 0)  # Cyan
        elif "Down" in gaze_direction:
            color = (0, 255, 0)    # Green
        elif "Center" in gaze_direction:
            color = (255, 255, 255)  # White
        elif "Closed" in gaze_direction:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)    # Red
        
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw additional info
        cv2.putText(frame, f"Left EAR: {results['left_ear']:.2f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Right EAR: {results['right_ear']:.2f}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Blinks: {results['blink_count']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

def main():
    """Main function with enhanced features"""
    print("🚀 Starting Enhanced Eye Movement Detection System")
    
    # Initialize detector
    detector = EnhancedEyeMovementDetector()
    
    # Open camera with better parameters
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("📹 Camera opened successfully")
    print("Controls: Press 'q' to quit, 'r' to reset blink counter")
    print("=" * 60)
    
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
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                print(f"📊 FPS: {fps:.1f}, Gaze: {results['gaze_direction']}, "
                      f"Blinks: {results['blink_count']}")
            
            # Display
            cv2.imshow('Enhanced Eye Movement Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.blink_counter = 0
                print("🔄 Blink counter reset")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()