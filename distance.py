# Enhanced Vehicle Distance Measurement System with Bird's Eye View
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# ==================== CONFIGURATION ====================

# ROI Zones for 2042 × 1148px resolution
ROI_ZONES = {
    'LEFT': np.array([[240, 600], [925, 550], [312, 1100], [100, 1100]], dtype=np.int32),
    'MAIN': np.array([[925, 550], [1025, 550], [1712, 1100], [312, 1100]], dtype=np.int32),
    'RIGHT': np.array([[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]], dtype=np.int32)
}

# Zone colors for visualization
ZONE_COLORS = {
    'LEFT': (255, 100, 100),   # Light Blue
    'MAIN': (100, 255, 100),   # Light Green
    'RIGHT': (100, 100, 255)   # Light Red
}

# Camera and measurement parameters
FOCAL_LENGTH = 500
OPTICAL_CENTERS = {'LEFT': (156, 1050), 'MAIN': (1000, 1020), 'RIGHT': (1868, 1050)}
TARGET_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Thresholds
CONFIDENCE_THRESHOLD = 0.7
LICENSE_PLATE_CONFIDENCE = 0.475
MAX_DISPLAY_DISTANCE = 15
WARNING_DISTANCES = {'LEFT': 1, 'MAIN': 2, 'RIGHT': 1}
DISTANCE_DISPLAY_THRESHOLDS = {'LEFT': 5, 'RIGHT': 5}
MAIN_ROI_WARNING_THRESHOLD = 5.0

# Bird's Eye View Configuration
BEV_CONFIG = {
    'width': 600,
    'height': 800,
    'scale': 30,  # pixels per meter
    'max_distance': 20,  # meters
    'ego_position': (300, 700)  # Position of ego vehicle on BEV
}

# ==================== VEHICLE TRACKING ====================

class VehicleTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.history = {}  # Store position history
        
    def register(self, centroid, bbox, distance, zone, class_id):
        self.objects[self.next_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'distance': distance,
            'zone': zone,
            'class_id': class_id
        }
        self.disappeared[self.next_id] = 0
        self.history[self.next_id] = deque(maxlen=30)
        self.history[self.next_id].append(centroid)
        self.next_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.history:
            del self.history[object_id]
            
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for det in detections:
                self.register(det['centroid'], det['bbox'], det['distance'], 
                            det['zone'], det['class_id'])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
            
            detection_centroids = [det['centroid'] for det in detections]
            
            # Compute distances between existing and new centroids
            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, dc in enumerate(detection_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))
            
            # Match objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 100:  # Max distance threshold
                    continue
                    
                object_id = object_ids[row]
                det = detections[col]
                self.objects[object_id] = {
                    'centroid': det['centroid'],
                    'bbox': det['bbox'],
                    'distance': det['distance'],
                    'zone': det['zone'],
                    'class_id': det['class_id']
                }
                self.history[object_id].append(det['centroid'])
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Register new detections
            unused_cols = set(range(len(detection_centroids))) - used_cols
            for col in unused_cols:
                det = detections[col]
                self.register(det['centroid'], det['bbox'], det['distance'], 
                            det['zone'], det['class_id'])
            
            # Mark disappeared objects
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects

# ==================== HELPER FUNCTIONS ====================

def is_point_in_roi(point, roi_coordinates):
    return cv2.pointPolygonTest(roi_coordinates, point, False) >= 0

def get_vehicle_roi_zone(center_point):
    for zone_name, roi_coords in ROI_ZONES.items():
        if is_point_in_roi(center_point, roi_coords):
            return zone_name
    return None

def calculate_distance(bbox, class_id, zone_name):
    if class_id not in REAL_VEHICLE_HEIGHTS or zone_name not in OPTICAL_CENTERS:
        return 0
        
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    if bbox_height <= 0:
        return 0
        
    vehicle_center_x, vehicle_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    optical_center_x, optical_center_y = OPTICAL_CENTERS[zone_name]
    displacement = np.sqrt((vehicle_center_x - optical_center_x)**2 + 
                          (vehicle_center_y - optical_center_y)**2)
    
    distance = (REAL_VEHICLE_HEIGHTS[class_id] * FOCAL_LENGTH) / bbox_height
    return distance * (1.0 + displacement * 0.0001)

def should_display_distance(distance, zone_name):
    threshold = DISTANCE_DISPLAY_THRESHOLDS.get(zone_name, 999)
    return distance <= threshold

def get_distance_color(distance, zone_name):
    warning_distance = WARNING_DISTANCES.get(zone_name, 2.0)
    
    if zone_name == 'MAIN' and distance < MAIN_ROI_WARNING_THRESHOLD:
        return (0, 0, 255)
    elif distance < warning_distance:
        return (0, 0, 255)
    else:
        return (0, 255, 0)

def blur_license_plates_in_vehicle(frame, vehicle_bbox, license_plate_model):
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    vehicle_region = frame[y1:y2, x1:x2]
    
    if vehicle_region.size == 0:
        return
    
    try:
        plate_results = license_plate_model(vehicle_region, conf=LICENSE_PLATE_CONFIDENCE, verbose=False)
        
        for plate_result in plate_results:
            if plate_result.boxes is not None:
                for plate_box in plate_result.boxes.xyxy.cpu().numpy():
                    px1, py1, px2, py2 = map(int, plate_box)
                    
                    frame_height, frame_width = frame.shape[:2]
                    px1, py1 = max(0, min(px1 + x1, frame_width)), max(0, min(py1 + y1, frame_height))
                    px2, py2 = max(0, min(px2 + x1, frame_width)), max(0, min(py2 + y1, frame_height))
                    
                    if px2 > px1 and py2 > py1:
                        plate_region = frame[py1:py2, px1:px2]
                        if plate_region.size > 0:
                            blurred_plate = cv2.GaussianBlur(plate_region, (51, 51), 30)
                            frame[py1:py2, px1:px2] = blurred_plate
                            
    except Exception as e:
        print(f"License plate detection error: {e}")

# ==================== DRAWING FUNCTIONS ====================

def draw_roi_zones(frame, show_labels=True):
    overlay = frame.copy()
    
    for zone_name, roi_coords in ROI_ZONES.items():
        color = ZONE_COLORS[zone_name]
        cv2.fillPoly(overlay, [roi_coords], color)
        cv2.polylines(frame, [roi_coords], True, color, 3)
        
        if show_labels:
            M = cv2.moments(roi_coords)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, zone_name, (cx - 40, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    return frame

def draw_distance_label(frame, bbox, distance, color, zone_name, object_id=None):
    x1, y1, x2, y2 = map(int, bbox)
    distance_text = f"{distance:.1f}m"
    if object_id is not None:
        distance_text = f"ID:{object_id} {distance_text}"
    
    warning_distance = WARNING_DISTANCES.get(zone_name, 2.0)
    
    if distance < warning_distance:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 0.8, 2
    (text_width, text_height), _ = cv2.getTextSize(distance_text, font, font_scale, thickness)
    
    frame_height, frame_width = frame.shape[:2]
    center_x = (x1 + x2) // 2
    text_x = max(10, min(center_x - text_width // 2, frame_width - text_width - 20))
    text_y = min(frame_height - 35, y2 + text_height + 20)
    
    padding = 8
    bg_x1 = max(0, text_x - padding)
    bg_y1 = max(0, text_y - text_height - padding)
    bg_x2 = min(frame_width, text_x + text_width + padding)
    bg_y2 = min(frame_height, text_y + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
    
    cv2.putText(frame, distance_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, distance_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def draw_warning_message(frame, message):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 1.2, 3
    
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    position_x, position_y = width//2 - text_width//2, 80
    
    padding = 15
    bg_x1 = max(0, position_x - padding)
    bg_y1 = max(0, position_y - text_height - padding)
    bg_x2 = min(width, position_x + text_width + padding)
    bg_y2 = min(height, position_y + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 3)
    
    cv2.putText(frame, message, (position_x, position_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, message, (position_x, position_y), font, font_scale, (255, 255, 255), thickness)

def draw_statistics_panel(frame, stats):
    """Draw statistics panel on frame"""
    height, width = frame.shape[:2]
    panel_width = 300
    panel_height = 200
    panel_x = width - panel_width - 20
    panel_y = 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_height = 30
    y = panel_y + 30
    
    cv2.putText(frame, "VEHICLE STATISTICS", (panel_x + 10, y), 
               font, 0.7, (255, 255, 255), 2)
    y += line_height
    
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (panel_x + 10, y), 
                   font, font_scale, (255, 255, 255), 1)
        y += line_height

# ==================== BIRD'S EYE VIEW ====================

def create_birds_eye_view(tracked_objects, frame_shape):
    """Create bird's eye view representation"""
    bev = np.zeros((BEV_CONFIG['height'], BEV_CONFIG['width'], 3), dtype=np.uint8)
    
    # Draw grid
    scale = BEV_CONFIG['scale']
    for i in range(0, BEV_CONFIG['height'], scale):
        cv2.line(bev, (0, i), (BEV_CONFIG['width'], i), (30, 30, 30), 1)
    for i in range(0, BEV_CONFIG['width'], scale):
        cv2.line(bev, (i, 0), (i, BEV_CONFIG['height']), (30, 30, 30), 1)
    
    # Draw distance markers
    ego_x, ego_y = BEV_CONFIG['ego_position']
    for dist in range(5, BEV_CONFIG['max_distance'], 5):
        y = ego_y - (dist * scale)
        if y > 0:
            cv2.line(bev, (0, y), (BEV_CONFIG['width'], y), (50, 50, 50), 1)
            cv2.putText(bev, f"{dist}m", (10, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Draw lanes (LEFT, MAIN, RIGHT zones)
    lane_width = BEV_CONFIG['width'] // 3
    cv2.line(bev, (lane_width, 0), (lane_width, BEV_CONFIG['height']), (100, 100, 100), 2)
    cv2.line(bev, (lane_width * 2, 0), (lane_width * 2, BEV_CONFIG['height']), (100, 100, 100), 2)
    
    # Label lanes
    cv2.putText(bev, "LEFT", (lane_width // 2 - 20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    cv2.putText(bev, "MAIN", (lane_width + lane_width // 2 - 20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    cv2.putText(bev, "RIGHT", (lane_width * 2 + lane_width // 2 - 30, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    
    # Draw ego vehicle
    cv2.circle(bev, BEV_CONFIG['ego_position'], 15, (0, 255, 0), -1)
    cv2.putText(bev, "EGO", (ego_x - 15, ego_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw detected vehicles
    for obj_id, obj_data in tracked_objects.items():
        distance = obj_data['distance']
        zone = obj_data['zone']
        class_id = obj_data['class_id']
        
        if distance > BEV_CONFIG['max_distance']:
            continue
        
        # Calculate BEV position
        zone_offset = {'LEFT': 0, 'MAIN': lane_width, 'RIGHT': lane_width * 2}
        bev_x = zone_offset.get(zone, lane_width) + lane_width // 2
        bev_y = ego_y - int(distance * scale)
        
        if 0 < bev_y < BEV_CONFIG['height']:
            # Color based on zone and distance
            color = ZONE_COLORS.get(zone, (255, 255, 255))
            if distance < WARNING_DISTANCES.get(zone, 2.0):
                color = (0, 0, 255)
            
            # Draw vehicle
            cv2.circle(bev, (bev_x, bev_y), 10, color, -1)
            cv2.circle(bev, (bev_x, bev_y), 10, (255, 255, 255), 2)
            
            # Draw ID and distance
            text = f"ID:{obj_id}"
            cv2.putText(bev, text, (bev_x + 15, bev_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(bev, f"{distance:.1f}m", (bev_x + 15, bev_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw line to ego vehicle
            cv2.line(bev, (bev_x, bev_y), BEV_CONFIG['ego_position'], color, 1)
    
    # Add title
    cv2.putText(bev, "BIRD'S EYE VIEW", (BEV_CONFIG['width']//2 - 80, BEV_CONFIG['height'] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return bev

# ==================== MAIN PROCESSING ====================

def main():
    # Load models
    print("Loading models...")
    model = YOLO(r"yolov10x.pt")
    license_plate_model = YOLO(r"Vehicle-Distance-Measurement-System-main\Vehicle-Distance-Measurement-System-main\license-plate.pt")
    
    # Initialize tracker
    tracker = VehicleTracker(max_disappeared=30)
    
    # Open video
    video_path = r"dm.mp4"
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_main = cv2.VideoWriter('output_main_view.mp4', fourcc, fps, (frame_width, frame_height))
    output_bev = cv2.VideoWriter('output_birds_eye.mp4', fourcc, fps, 
                                 (BEV_CONFIG['width'], BEV_CONFIG['height']))
    output_combined = cv2.VideoWriter('output_combined.mp4', fourcc, fps, 
                                     (frame_width + BEV_CONFIG['width'], frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    print("Processing video...")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Draw ROI zones
        annotated_frame = draw_roi_zones(annotated_frame, show_labels=True)
        
        # Run detection
        results = model(frame, classes=TARGET_CLASSES, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Blur license plates
                    blur_license_plates_in_vehicle(annotated_frame, bbox, license_plate_model)
                    
                    zone_name = get_vehicle_roi_zone(center_point)
                    if zone_name is None:
                        continue
                    
                    distance = calculate_distance(bbox, class_id, zone_name)
                    
                    if distance <= MAX_DISPLAY_DISTANCE:
                        detections.append({
                            'centroid': center_point,
                            'bbox': bbox,
                            'distance': distance,
                            'zone': zone_name,
                            'class_id': class_id
                        })
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Draw tracked vehicles
        warning_active = False
        stats = {'Total': len(tracked_objects), 'LEFT': 0, 'MAIN': 0, 'RIGHT': 0}
        
        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            distance = obj_data['distance']
            zone = obj_data['zone']
            
            stats[zone] += 1
            
            if should_display_distance(distance, zone):
                color = get_distance_color(distance, zone)
                draw_distance_label(annotated_frame, bbox, distance, color, zone, obj_id)
            
            warning_distance = WARNING_DISTANCES.get(zone, 2.0)
            if distance < warning_distance:
                warning_active = True
        
        # Draw warning if needed
        if warning_active:
            draw_warning_message(annotated_frame, "WARNING: VEHICLE VERY CLOSE!")
        
        # Draw statistics
        draw_statistics_panel(annotated_frame, stats)
        
        # Create bird's eye view
        bev_frame = create_birds_eye_view(tracked_objects, frame.shape)
        
        # Resize BEV to match frame height
        bev_resized = cv2.resize(bev_frame, (BEV_CONFIG['width'], frame_height))
        
        # Create combined view
        combined_frame = np.hstack([annotated_frame, bev_resized])
        
        # Write outputs
        output_main.write(annotated_frame)
        output_bev.write(bev_frame)
        output_combined.write(combined_frame)
        
        # Display
        display_scale = 0.6
        display_combined = cv2.resize(combined_frame, 
                                     (int(combined_frame.shape[1] * display_scale), 
                                      int(combined_frame.shape[0] * display_scale)))
        cv2.imshow('Vehicle Distance Measurement - Combined View', display_combined)
        
        # Progress update
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {fps_actual:.1f}")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    video_capture.release()
    output_main.release()
    output_bev.release()
    output_combined.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}")
    print(f"\nOutput files created:")
    print(f"  - output_main_view.mp4")
    print(f"  - output_birds_eye.mp4")
    print(f"  - output_combined.mp4")

if __name__ == "__main__":
    main()