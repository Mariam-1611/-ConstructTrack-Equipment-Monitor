import cv2
import numpy as np
from collections import deque


class ActivityClassifier:
    """
    Classifies what a piece of equipment is currently doing:
        - DIGGING    : arm moving downward repeatedly
        - SWINGING   : arm moving left or right (loading/unloading)
        - DUMPING    : arm moving upward and outward
        - WAITING    : machine is active but arm barely moving

    HOW IT WORKS:
    We use Optical Flow (Lucas-Kanade method) on the UPPER region
    of the bounding box (where the arm is).

    Optical Flow tracks how pixels move between two frames:
        - It gives us vectors (dx, dy) for each tracked point
        - dy negative = moving UP
        - dy positive = moving DOWN
        - dx = moving LEFT or RIGHT
    """

    def __init__(self, history_size=8):
       
        self.history_size = history_size

      
        self.motion_history = {}

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),       
            maxLevel=2,             
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            )
        )

        # Feature detection 
        self.feature_params = dict(
            maxCorners=30,         
            qualityLevel=0.3,       
            minDistance=7,          
            blockSize=7
        )

        # Store previous grayscale upper regions per equipment
        self.prev_upper_gray = {}

    def classify(self, equipment_id, frame, bbox, is_active):
        """
        Main method — call every frame for each detected equipment.
        Returns:
            activity string: "DIGGING", "SWINGING", "DUMPING", or "WAITING"
        """
        if not is_active:
            return "WAITING"

        x1, y1, x2, y2 = bbox

        # Safety check
        if x2 <= x1 or y2 <= y1:
            return "WAITING"

        # Crop the equipment region
        equipment_crop = frame[y1:y2, x1:x2]
        if equipment_crop.size == 0:
            return "WAITING"

        h, w = equipment_crop.shape[:2]
        mid_y = h // 2

        # We only care about the UPPER region (arm area)
        upper_region = equipment_crop[0:mid_y, 0:w]
        curr_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)

        # Initialize history for new equipment
        if equipment_id not in self.motion_history:
            self.motion_history[equipment_id] = deque(maxlen=self.history_size)
            self.prev_upper_gray[equipment_id] = curr_gray.copy()
            return "WAITING"

        prev_gray = self.prev_upper_gray[equipment_id]

        # Resize prev if needed 
        if prev_gray.shape != curr_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))

  
        # Find good feature points to track in previous frame
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=None,
            **self.feature_params
        )

        if prev_points is None or len(prev_points) < 3:
            # Not enough features to track — update and return waiting
            self.prev_upper_gray[equipment_id] = curr_gray.copy()
            return "WAITING"

        # Track those points in the current frame
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            **self.lk_params
        )

        # Keep only successfully tracked points
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) < 3:
            self.prev_upper_gray[equipment_id] = curr_gray.copy()
            return "WAITING"

        # Calculate motion vectors 
        vectors = good_curr - good_prev  

        # Average motion vector across all tracked points
        avg_dx = float(np.mean(vectors[:, 0]))  # horizontal movement
        avg_dy = float(np.mean(vectors[:, 1]))  # vertical movement

        # Store in history
        self.motion_history[equipment_id].append((avg_dx, avg_dy))

        # Update previous frame
        self.prev_upper_gray[equipment_id] = curr_gray.copy()

        # Classify
        return self._classify_from_history(equipment_id)

    def _classify_from_history(self, equipment_id):
        """
        Looks at the last N frames of motion vectors and
        decides what activity is happening.

        Rules:
            avg_dy strongly negative (arm going UP)      → DUMPING
            avg_dy strongly positive (arm going DOWN)    → DIGGING
            avg_dx dominant (arm going sideways)         → SWINGING
            motion present but no clear direction        → WAITING
        """
        history = list(self.motion_history[equipment_id])

        if len(history) < 3:
            return "WAITING"

        # Average over recent history
        avg_dx = np.mean([v[0] for v in history])
        avg_dy = np.mean([v[1] for v in history])

        abs_dx = abs(avg_dx)
        abs_dy = abs(avg_dy)

        # Minimum movement to classify (filter noise)
        min_movement = 0.8

        if abs_dx < min_movement and abs_dy < min_movement:
            # Very little movement even though active
            return "WAITING"

        # Decide dominant direction
        if abs_dy > abs_dx:
            # Vertical motion dominates
            if avg_dy > 0:
                # Moving DOWN → arm digging into ground
                return "DIGGING"
            else:
                # Moving UP → arm lifting/dumping load
                return "DUMPING"
        else:
            # Horizontal motion dominates → swinging left or right
            return "SWINGING"

    def reset(self, equipment_id=None):
        """Reset stored history for one or all equipment."""
        if equipment_id:
            self.motion_history.pop(equipment_id, None)
            self.prev_upper_gray.pop(equipment_id, None)
        else:
            self.motion_history.clear()
            self.prev_upper_gray.clear() 
