import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from scipy.spatial.distance import cosine
import pytesseract
from scenedetect import detect, ContentDetector
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import json
from collections import defaultdict

# Configuration
HIGHLIGHT_PATH = "./highlights/15sec_input_720p.mp4"
HOMOGRAPHY_DIR = "./highlights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL = "yolov8n.pt"
REID_MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
FEATURE_THRESHOLD = 0.25  # More strict threshold for matching
JERSEY_CONFIDENCE = 0.7
MIN_TRACK_LENGTH = 5  # Minimum frames to confirm a track
MAX_MISSED_FRAMES = 20  # Maximum frames to keep a track without updates
pytesseract.pytesseract.tesseract_cmd = r'./highlights/tesseract.exe'
OUTPUT_PERFORMANCE_FILE = "./highlights/performance_metrics.json"

# Initialize DeepSORT tracker with adjusted parameters
tracker = DeepSort(
    max_age=MAX_MISSED_FRAMES,
    n_init=MIN_TRACK_LENGTH,
    nn_budget=100,
    max_cosine_distance=FEATURE_THRESHOLD
)

# Performance tracking class
class PerformanceTracker:
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.feature_times = []
        self.jersey_times = []
        self.tracking_times = []
        self.detections_per_frame = []
        self.jersey_success = 0
        self.jersey_attempts = 0
        self.track_switches = 0
        self.lost_tracks = 0
        self.active_tracks = defaultdict(int)  # Track ID to frame count
        self.previous_track_ids = set()

    def update_frame_time(self, time_taken):
        self.frame_times.append(time_taken)

    def update_component_times(self, det_time, feat_time, jersey_time, track_time):
        self.detection_times.append(det_time)
        self.feature_times.append(feat_time)
        self.jersey_times.append(jersey_time)
        self.tracking_times.append(track_time)

    def update_detections(self, num_detections):
        self.detections_per_frame.append(num_detections)

    def update_jersey_detection(self, success):
        self.jersey_attempts += 1
        if success:
            self.jersey_success += 1

    def update_tracking_metrics(self, current_track_ids):
        # Count track switches (new tracks appearing)
        new_tracks = current_track_ids - self.previous_track_ids
        self.track_switches += len(new_tracks)
        
        # Update active tracks duration
        for tid in current_track_ids:
            self.active_tracks[tid] += 1
        
        # Count lost tracks
        lost = self.previous_track_ids - current_track_ids
        self.lost_tracks += len(lost)
        
        self.previous_track_ids = current_track_ids

    def save_metrics(self, output_file):
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        metrics = {
            "total_frames": len(self.frame_times),
            "average_fps": 1.0 / avg_frame_time if avg_frame_time > 0 else 0,
            "average_detection_time_ms": np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            "average_feature_extraction_time_ms": np.mean(self.feature_times) * 1000 if self.feature_times else 0,
            "average_jersey_detection_time_ms": np.mean(self.jersey_times) * 1000 if self.jersey_times else 0,
            "average_tracking_time_ms": np.mean(self.tracking_times) * 1000 if self.tracking_times else 0,
            "average_detections_per_frame": np.mean(self.detections_per_frame) if self.detections_per_frame else 0,
            "jersey_detection_success_rate": (self.jersey_success / self.jersey_attempts) if self.jersey_attempts > 0 else 0,
            "total_track_switches": self.track_switches,
            "total_lost_tracks": self.lost_tracks,
            "average_track_duration_frames": np.mean(list(self.active_tracks.values())) if self.active_tracks else 0
        }
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics

    def print_summary(self):
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        print("\nPerformance Summary:")
        print(f"Total Frames Processed: {len(self.frame_times)}")
        print(f"Average FPS: {1.0 / avg_frame_time:.2f}" if avg_frame_time > 0 else "Average FPS: N/A")
        print(f"Average Detection Time: {np.mean(self.detection_times) * 1000:.2f} ms" if self.detection_times else "Average Detection Time: N/A")
        print(f"Average Feature Extraction Time: {np.mean(self.feature_times) * 1000:.2f} ms" if self.feature_times else "Average Feature Extraction Time: N/A")
        print(f"Average Jersey Detection Time: {np.mean(self.jersey_times) * 1000:.2f} ms" if self.jersey_times else "Average Jersey Detection Time: N/A")
        print(f"Average Tracking Time: {np.mean(self.tracking_times) * 1000:.2f} ms" if self.tracking_times else "Average Tracking Time: N/A")
        print(f"Average Detections per Frame: {np.mean(self.detections_per_frame):.2f}" if self.detections_per_frame else "Average Detections per Frame: N/A")
        print(f"Jersey Detection Success Rate: {(self.jersey_success / self.jersey_attempts) * 100:.2f}%" if self.jersey_attempts > 0 else "Jersey Detection Success Rate: N/A")
        print(f"Total Track Switches: {self.track_switches}")
        print(f"Total Lost Tracks: {self.lost_tracks}")
        print(f"Average Track Duration: {np.mean(list(self.active_tracks.values())):.2f} frames" if self.active_tracks else "Average Track Duration: N/A")

# Preprocess image for Re-ID model
def preprocess_image(image, size=(128, 128)):
    image = cv2.resize(image, size)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return image

# Extract Re-ID features
def extract_features(model, images):
    features = []
    with torch.no_grad():
        for img in images:
            img_tensor = preprocess_image(img)
            feat = model(img_tensor).cpu().numpy().reshape(-1)
            features.append(feat)
    return np.array(features) if features else np.array([], dtype=np.float32)

# Detect jersey number with improved preprocessing
def detect_jersey_number(image):
    try:
        # Enhanced preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # OCR with optimized parameters
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        number13 = ''.join([c for c in text if c.isdigit()])
        return number13 if number13 else None
    except Exception as e:
        print(f"Jersey number detection error: {str(e)}")
        return None

# Load homography
def load_homography(angle_idx):
    homography_path = f"{HOMOGRAPHY_DIR}/homography_angle{angle_idx}.npy"
    if Path(homography_path).exists():
        return np.load(homography_path)
    print(f"Warning: Homography for angle {angle_idx} not found. Using identity matrix.")
    return np.eye(3)

# Detect scenes
def detect_scenes(video_path):
    scene_list = detect(video_path, ContentDetector(threshold=30.0))
    return [(start.get_frames(), end.get_frames()) for start, end in scene_list]

# Calculate average feature vector for better matching
def get_average_feature(feature_history):
    if len(feature_history) == 0:
        return None
    return np.mean(feature_history, axis=0)

# Main pipeline with improved ID stability and performance analysis
def main():
    cap = cv2.VideoCapture(HIGHLIGHT_PATH)
    if not cap.isOpened():
        print("Error: Could not open highlight video.")
        return

    scene_boundaries = detect_scenes(HIGHLIGHT_PATH)
    print(f"Detected {len(scene_boundaries)} scenes: {scene_boundaries}")

    detector = YOLO(YOLO_MODEL)
    
    # Enhanced player database structure
    global_player_db = {}  # {track_id: {
                         #    "features": [], 
                         #    "jersey": str, 
                         #    "last_frame": int,
                         #    "stable_id": int,
                         #    "first_frame": int,
                         #    "confirmed": bool}
                         # }
    
    stable_id_counter = 0  # Counter for unique, stable IDs
    frame_idx = 0
    scene_idx = 0
    perf_tracker = PerformanceTracker()

    while cap.isOpened():
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # Scene boundary management
        while scene_idx < len(scene_boundaries) and frame_idx > scene_boundaries[scene_idx][1]:
            scene_idx += 1
        if scene_idx >= len(scene_boundaries):
            break
        current_scene = scene_idx + 1
        H = load_homography(current_scene)

        # Object detection
        det_start = time.time()
        results = detector(frame, conf=0.2)  # Lowered confidence
        boxes = results[0].boxes
        if boxes is not None:
            boxes = boxes[boxes.cls == 0].xyxy.cpu().numpy()  # Filter for person class (0)
        else:
            boxes = np.array([], dtype=np.float32)
        det_time = time.time() - det_start

        # Extract detections and features
        feat_start = time.time()
        images = []
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if x2 > x1 and y2 > y1 and 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:
                img = frame[y1:y2, x1:x2]
                if img.size > 0:
                    images.append(img)
                    bboxes.append([x1, y1, x2, y2])
        features = extract_features(REID_MODEL, images)
        feat_time = time.time() - feat_start

        perf_tracker.update_detections(len(features))
        print(f"Frame {frame_idx}: {len(features)} detections")

        # Prepare detections for tracking
        jersey_time = 0
        detections = []
        for i, (bbox, feat) in enumerate(zip(bboxes, features)):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            jersey_start = time.time()
            jersey = detect_jersey_number(images[i])
            jersey_time += time.time() - jersey_start
            perf_tracker.update_jersey_detection(jersey is not None)
            detections.append(([x1, y1, w, h], 0.99, i))  # Store index to link back onboarding

        # Update tracker
        track_start = time.time()
        tracks = tracker.update_tracks(detections, frame=frame)
        track_time = time.time() - track_start

        # Assign IDs and update database
        active_track_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue  # Only process confirmed tracks
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            if ltrb is None:
                continue
                
            x1, y1, x2, y2 = map(int, ltrb)
            det_idx = track.detection_index if hasattr(track, 'detection_index') else None
            
            # Get features and jersey if available
            feat = None
            jersey = None
            if det_idx is not None and 0 <= det_idx < len(features):
                feat = features[det_idx]
                jersey_start = time.time()
                jersey = detect_jersey_number(images[det_idx]) if det_idx < len(images) else None
                jersey_time += time.time() - jersey_start
                perf_tracker.update_jersey_detection(jersey is not None)

            # Track management logic
            if track_id not in global_player_db:
                # Try to find matching existing player
                best_match_id = None
                best_dist = FEATURE_THRESHOLD
                
                for db_id, data in global_player_db.items():
                    # Skip recently updated tracks to avoid duplicate assignments
                    if frame_idx - data["last_frame"] < 10:
                        continue
                        
                    db_feat = get_average_feature(data["features"][-5:])  # Use average of last 5 features
                    if feat is not None and db_feat is not None:
                        dist = cosine(feat, db_feat)
                        
                        # Additional jersey matching boost
                        if jersey and data["jersey"] and jersey == data["jersey"]:
                            dist *= 0.25  # Strong boost for jersey match
                            
                        if dist < best_dist:
                            best_dist = dist
                            best_match_id = db_id
                
                # Assign to existing player or create new entry
                if best_match_id is not None:
                    # Update existing entry with most recent info
                    global_player_db[best_match_id]["features"].append(feat)
                    global_player_db[best_match_id]["last_frame"] = frame_idx
                    if jersey:
                        global_player_db[best_match_id]["jersey"] = jersey
                        
                    # Maintain the same stable ID
                    track.stable_id = global_player_db[best_match_id]["stable_id"]
                else:
                    # Initialize new player entry
                    global_player_db[track_id] = {
                        "features": [feat] if feat is not None else [],
                        "jersey": jersey,
                        "last_frame": frame_idx,
                        "stable_id": stable_id_counter,
                        "first_frame": frame_idx,
                        "confirmed": True
                    }
                    track.stable_id = stable_id_counter
                    stable_id_counter += 1
            else:
                # Update existing track
                if feat is not None:
                    global_player_db[track_id]["features"].append(feat)
                global_player_db[track_id]["last_frame"] = frame_idx
                if jersey:
                    global_player_db[track_id]["jersey"] = jersey
                    
                # Carry forward the stable ID
                track.stable_id = global_player_db[track_id]["stable_id"]
                
            active_track_ids.add(track_id)

        # Update tracking metrics
        perf_tracker.update_tracking_metrics(active_track_ids)

        # Clean up stale tracks
        stale_tracks = [tid for tid, data in global_player_db.items() 
                        if frame_idx - data["last_frame"] > MAX_MISSED_FRAMES]
        for tid in stale_tracks:
            del global_player_db[tid]

        # Visualization
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            ltrb = track.to_ltrb()
            if ltrb is None:
                continue
                
            x1, y1, x2, y2 = map(int, ltrb)
            stable_id = track.stable_id if hasattr(track, 'stable_id') else track.track_id
            
            # Get player info from database
            player_info = global_player_db.get(track.track_id, {})
            jersey = player_info.get("jersey", None)
            
            # Draw bounding box and info
            color = (0, 255, 0)  # Green for confirmed tracks
            label = f"ID: {stable_id}"
            if jersey:
                label += f" ({jersey})"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw field position if homography is available
            if not np.array_equal(H, np.eye(3)):
                center = np.array([[(x1 + x2)/2, (y1 + y2)/2]], dtype=np.float32)
                field_pos = cv2.perspectiveTransform(center[None, :, :], H)[0][0]
                cv2.putText(frame, f"({int(field_pos[0])},{int(field_pos[1])})", 
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame with player tracking
        cv2.imshow("Player Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update performance metrics
        frame_time = time.time() - frame_start_time
        perf_tracker.update_frame_time(frame_time)
        perf_tracker.update_component_times(det_time, feat_time, jersey_time, track_time)
            
        frame_idx += 1

    # Save and display performance metrics
    metrics = perf_tracker.save_metrics(OUTPUT_PERFORMANCE_FILE)
    perf_tracker.print_summary()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()