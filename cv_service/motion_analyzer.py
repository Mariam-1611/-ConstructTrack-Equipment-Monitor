import cv2
import numpy as np


class MotionAnalyzer:
    """
    Analyzes motion within a detected equipment bounding box.

    The KEY idea: instead of looking at the whole bounding box as one unit,
    we split it into TWO regions:
        - Upper region (top 50%): where the ARM and BUCKET are
        - Lower region (bottom 50%): where the TRACKS and BODY are

    This lets us detect arm-only motion correctly:
        - Upper moving + Lower still  → ACTIVE (arm_only)
        - Both moving                 → ACTIVE (full_body)
        - Both still                  → INACTIVE
    """

    def __init__(self, motion_threshold=500):
        self.motion_threshold = motion_threshold
        self.prev_frames = {}

    def _get_region_motion(self, prev_region, curr_region):
        prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_region, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # (motion score)
        motion_score = np.sum(thresh > 0)

        return motion_score

    def analyze(self, equipment_id, frame, bbox):
        """
        Main method — call this every frame for each detected equipment.
        """
        x1, y1, x2, y2 = bbox

        # Safety check: make sure bbox is valid
        if x2 <= x1 or y2 <= y1:
            return self._inactive_result()

        # Crop the equipment region from the full frame
        equipment_crop = frame[y1:y2, x1:x2]

        # Make sure crop is not empty
        if equipment_crop.size == 0:
            return self._inactive_result()

        h, w = equipment_crop.shape[:2]
        mid_y = h // 2  

        # Split into upper (arm area) and lower (tracks area)
        upper_region = equipment_crop[0:mid_y, 0:w]
        lower_region = equipment_crop[mid_y:h, 0:w]

        # First frame for this equipment — just save and return inactive
        if equipment_id not in self.prev_frames:
            self.prev_frames[equipment_id] = {
                "upper": upper_region.copy(),
                "lower": lower_region.copy()
            }
            return self._inactive_result()

        # Get previous regions
        prev = self.prev_frames[equipment_id]

        # Resize previous regions to match current (in case bbox shifted slightly)
        prev_upper = cv2.resize(prev["upper"], (upper_region.shape[1], upper_region.shape[0]))
        prev_lower = cv2.resize(prev["lower"], (lower_region.shape[1], lower_region.shape[0]))

        # Calculate motion scores for each region
        upper_score = self._get_region_motion(prev_upper, upper_region)
        lower_score = self._get_region_motion(prev_lower, lower_region)

        # Save current as previous for next frame
        self.prev_frames[equipment_id] = {
            "upper": upper_region.copy(),
            "lower": lower_region.copy()
        }

        # Decision Logic 
        upper_moving = upper_score > self.motion_threshold
        lower_moving = lower_score > self.motion_threshold

        if upper_moving and lower_moving:
            return {
                "is_active": True,
                "motion_source": "full_body",
                "upper_score": int(upper_score),
                "lower_score": int(lower_score)
            }
        elif upper_moving and not lower_moving:
            # Only the arm is moving — this is the KEY case the company wants
            return {
                "is_active": True,
                "motion_source": "arm_only",
                "upper_score": int(upper_score),
                "lower_score": int(lower_score)
            }
        else:
            # Nothing is moving
            return {
                "is_active": False,
                "motion_source": "stationary",
                "upper_score": int(upper_score),
                "lower_score": int(lower_score)
            }

    def _inactive_result(self):
        """Returns a default inactive result."""
        return {
            "is_active": False,
            "motion_source": "stationary",
            "upper_score": 0,
            "lower_score": 0
        }

    def reset(self, equipment_id=None):
        """
        Reset stored frames.
        Call with equipment_id to reset one machine,
        or no args to reset all.
        """
        if equipment_id:
            self.prev_frames.pop(equipment_id, None)
        else:
            self.prev_frames.clear() 
