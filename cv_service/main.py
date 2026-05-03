import cv2
import os
import time
import logging
from ultralytics import YOLO
from motion_analyzer import MotionAnalyzer
from activity_classifier import ActivityClassifier
from kafka_producer import EquipmentKafkaProducer

# Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
VIDEO_PATH   = os.getenv("VIDEO_PATH",   "./videos/video1.mp4")

# YOLO classes 
# 0=person, 2=car, 7=truck, 9=traffic light ...
EQUIPMENT_CLASSES = {"excavator", "truck", "bulldozer", "crane", "machinery"}

# Frames per 
FPS = 30.0

# Time Tracker 
class TimeTracker:
    """
    Tracks how long each equipment has been active vs idle.
    Updated every frame based on ACTIVE/INACTIVE state.
    """
    def __init__(self):
        # equipment_id → { tracked, active, idle }
        self.data = {}

    def update(self, equipment_id, is_active, seconds_per_frame):
        """Call this every frame for each detected equipment."""
        if equipment_id not in self.data:
            self.data[equipment_id] = {
                "total_tracked_seconds": 0.0,
                "total_active_seconds":  0.0,
                "total_idle_seconds":    0.0,
            }

        d = self.data[equipment_id]
        d["total_tracked_seconds"] += seconds_per_frame

        if is_active:
            d["total_active_seconds"] += seconds_per_frame
        else:
            d["total_idle_seconds"] += seconds_per_frame

    def get_analytics(self, equipment_id):
        """Returns time analytics dict including utilization percentage."""
        if equipment_id not in self.data:
            return {
                "total_tracked_seconds": 0.0,
                "total_active_seconds":  0.0,
                "total_idle_seconds":    0.0,
                "utilization_percent":   0.0
            }

        d = self.data[equipment_id]
        tracked = d["total_tracked_seconds"]

        utilization = (
            round((d["total_active_seconds"] / tracked) * 100, 1)
            if tracked > 0 else 0.0
        )

        return {
            "total_tracked_seconds": round(tracked, 2),
            "total_active_seconds":  round(d["total_active_seconds"], 2),
            "total_idle_seconds":    round(d["total_idle_seconds"], 2),
            "utilization_percent":   utilization
        }


# Equipment ID Generator 
def generate_equipment_id(track_id, label):
    """
    Generates a readable ID like EX-001 or TR-002
    from YOLO's numeric track ID.
    """
    prefix_map = {
        "excavator":  "EX",
        "truck":      "TR",
        "bulldozer":  "BU",
        "crane":      "CR",
        "machinery":  "MC",
    }
    prefix = prefix_map.get(label.lower(), "EQ")
    return f"{prefix}-{str(track_id).zfill(3)}"


# Draw Overlays on Frame 
def draw_overlay(frame, bbox, equipment_id, state, activity, analytics):
    """
    Draws bounding box and status info on the video frame.
    Green = ACTIVE, Red = INACTIVE
    """
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Split line showing upper/lower regions
    mid_y = (y1 + y2) // 2
    cv2.line(frame, (x1, mid_y), (x2, mid_y), (255, 255, 0), 1)

    # Labels
    label_lines = [
        f"{equipment_id}",
        f"{state} | {activity}",
        f"Util: {analytics['utilization_percent']}%",
        f"Active: {analytics['total_active_seconds']:.1f}s",
    ]

    for i, line in enumerate(label_lines):
        y_pos = max(y1 - 10 - (i * 18), 10)
        cv2.putText(
            frame, line,
            (x1, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA
        )

    return frame


#  Main Pipeline 
def main():
    logger.info("Starting Eagle Vision CV Service")
    logger.info(f"Video: {VIDEO_PATH}")
    logger.info(f"Kafka: {KAFKA_BROKER}")

    # Load YOLO model
    logger.info("Loading YOLO model...")
    model = YOLO("yolov8n.pt") 
    logger.info("YOLO model loaded!")

    #  Initialize components 
    motion_analyzer     = MotionAnalyzer(motion_threshold=500)
    activity_classifier = ActivityClassifier(history_size=8)
    time_tracker        = TimeTracker()
    producer            = EquipmentKafkaProducer(
                            bootstrap_servers=KAFKA_BROKER,
                            topic="equipment-detections"
                          )

    #  Open video 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {VIDEO_PATH}")
        return

    # Get actual FPS from video file
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    seconds_per_frame = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {fps:.1f} FPS, {total_frames} total frames")

    frame_id = 0

    # Frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video finished.")
            break

        frame_id += 1

        # Run YOLO tracking (track_id persists across frames)
        # persist=True keeps track IDs consistent frame to frame
        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is None:
            continue

        boxes   = results[0].boxes
        tracked_ids = boxes.id          
        class_ids   = boxes.cls        
        xyxy        = boxes.xyxy        

        if tracked_ids is None:
            continue

        # Process each detected object
        for i in range(len(tracked_ids)):
            track_id     = int(tracked_ids[i].item())
            class_id     = int(class_ids[i].item())
            label        = model.names[class_id].lower()
            bbox_tensor  = xyxy[i]

            # Convert bbox to integers
            x1 = int(bbox_tensor[0].item())
            y1 = int(bbox_tensor[1].item())
            x2 = int(bbox_tensor[2].item())
            y2 = int(bbox_tensor[3].item())
            bbox = (x1, y1, x2, y2)

            # Generate readable equipment ID
            equipment_id = generate_equipment_id(track_id, label)

            # ── Step 1: Motion Analysis ──
            motion_result = motion_analyzer.analyze(equipment_id, frame, bbox)
            is_active     = motion_result["is_active"]
            motion_source = motion_result["motion_source"]

            # ── Step 2: Activity Classification ──
            activity = activity_classifier.classify(
                equipment_id, frame, bbox, is_active
            )

            # ── Step 3: Time Tracking ──
            time_tracker.update(equipment_id, is_active, seconds_per_frame)
            analytics = time_tracker.get_analytics(equipment_id)

            # ── Step 4: Send to Kafka ──
            producer.send_detection(
                frame_id        = frame_id,
                equipment_id    = equipment_id,
                equipment_class = label,
                current_state   = "ACTIVE" if is_active else "INACTIVE",
                current_activity= activity,
                motion_source   = motion_source,
                time_analytics  = analytics
            )

            # ── Step 5: Draw on frame ──
            frame = draw_overlay(
                frame, bbox, equipment_id,
                "ACTIVE" if is_active else "INACTIVE",
                activity, analytics
            )

        # Log progress every 30 frames
        if frame_id % 30 == 0:
            logger.info(f"Processed frame {frame_id}/{total_frames}")

        # Optional: show video window (comment out in Docker)

    # ── Cleanup ──
    cap.release()
    producer.flush()
    producer.close()
    cv2.destroyAllWindows()
    logger.info("CV Service finished.")


if __name__ == "__main__":
    main() 
