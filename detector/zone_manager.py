import logging
from typing import List, Tuple, Dict, Optional
import math

logger = logging.getLogger(__name__)

class DynamicZoneManager:
    """
    Dynamically identifies if the camera geometry is Side-By-Side (Left/Right)
    or Top-And-Bottom based on a stability calibration phase.
    """
    
    def __init__(self, calibration_frames: int = 60, stability_threshold_px: float = 30.0):
        self.calibration_frames = calibration_frames
        self.stability_threshold_px = stability_threshold_px
        
        self.is_calibrated = False
        self.layout_type = None  # 'SIDE_BY_SIDE' or 'TOP_BOTTOM'
        self.split_val = 0.0
        
        # Calibration state
        self._consecutive_stable_frames = 0
        self._history = []  # List of pairs of center coords [(cx1, cy1), (cx2, cy2)]
        
    def _center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, person_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """
        Feed person bounding boxes every frame. Returns True if calibrated.
        """
        if self.is_calibrated:
            return True
            
        # We need exactly 2 people to calibrate
        if len(person_boxes) < 2:
            return False
            
        # Get the two largest boxes
        sorted_boxes = sorted(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        top_2 = sorted_boxes[:2]
        
        c1 = self._center(top_2[0])
        c2 = self._center(top_2[1])
        
        # Sort centers by X just to keep them paired roughly
        centers = sorted([c1, c2], key=lambda c: c[0])
        self._history.append(centers)
        
        # Instead of requiring them to be perfectly still, just collect 30 valid frames
        if len(self._history) >= 30:
            self._finalize_calibration()
            
        return self.is_calibrated
        
    def _finalize_calibration(self):
        import numpy as np
        
        # self._history is a list of 30 pairs: [ [(x1,y1), (x2,y2)], ... ]
        # Because we sorted by X, list 1 is generally the left-most person, list 2 is the right-most.
        c1_list = [pair[0] for pair in self._history]
        c2_list = [pair[1] for pair in self._history]
        
        # Use median to ignore outliers (e.g., someone walking by)
        med_c1_x = np.median([c[0] for c in c1_list])
        med_c1_y = np.median([c[1] for c in c1_list])
        med_c2_x = np.median([c[0] for c in c2_list])
        med_c2_y = np.median([c[1] for c in c2_list])
        
        dist_x = abs(med_c1_x - med_c2_x)
        dist_y = abs(med_c1_y - med_c2_y)
        
        if dist_x > dist_y:
            self.layout_type = 'SIDE_BY_SIDE'
            self.split_val = (med_c1_x + med_c2_x) / 2.0
            print(f"\n[ZoneManager] Calibrated: SIDE_BY_SIDE with vertical split at x={self.split_val:.1f}")
            logger.info(f"[ZoneManager] Calibrated: SIDE_BY_SIDE with vertical split at x={self.split_val:.1f}")
        else:
            self.layout_type = 'TOP_BOTTOM'
            self.split_val = (med_c1_y + med_c2_y) / 2.0
            print(f"\n[ZoneManager] Calibrated: TOP_BOTTOM with horizontal split at y={self.split_val:.1f}")
            logger.info(f"[ZoneManager] Calibrated: TOP_BOTTOM with horizontal split at y={self.split_val:.1f}")
            
        self.is_calibrated = True
        
    def assign_pilots(self, person_boxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Assigns boxes to pid=1 and pid=2 based on the calibrated layout.
        Uses relative sorting for the two largest boxes to be robust against shifting.
        Falls back to absolute split_val if only 1 box is present.
        """
        if not self.is_calibrated or not person_boxes:
            return {}
            
        area = lambda b: (b[2]-b[0])*(b[3]-b[1])
        top_boxes = sorted(person_boxes, key=area, reverse=True)[:2]
        
        result = {}
        
        if len(top_boxes) == 2:
            # We have 2 people, assign them relatively!
            if self.layout_type == 'SIDE_BY_SIDE':
                # Left-most is Pilot 1, Right-most is Pilot 2
                top_boxes.sort(key=lambda b: self._center(b)[0])
                result[1] = top_boxes[0]
                result[2] = top_boxes[1]
            else:
                # Top-most is Pilot 2, Bottom-most is Pilot 1
                top_boxes.sort(key=lambda b: self._center(b)[1])
                result[2] = top_boxes[0]
                result[1] = top_boxes[1]
        else:
            # Only 1 person detected, we MUST use the absolute split line to guess who it is
            box = top_boxes[0]
            cx, cy = self._center(box)
            if self.layout_type == 'SIDE_BY_SIDE':
                if cx < self.split_val:
                    result[1] = box
                else:
                    result[2] = box
            else:
                if cy < self.split_val:
                    result[2] = box
                else:
                    result[1] = box
                    
        return result
        
    def get_zone(self, pid: int, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Returns the geographic (x1, y1, x2, y2) bounds of the assigned zone for drawing.
        """
        if not self.is_calibrated:
            return (0, 0, frame_width, frame_height)
            
        split = int(self.split_val)
        if self.layout_type == 'SIDE_BY_SIDE':
            if pid == 1:
                return (0, 0, split, frame_height)
            else:
                return (split, 0, frame_width, frame_height)
        else:
            if pid == 2:
                return (0, 0, frame_width, split)
            else:
                return (0, split, frame_width, frame_height)

