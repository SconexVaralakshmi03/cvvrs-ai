import logging
from typing import List, Tuple, Dict, Optional
import math
import numpy as np

from detector.enums import CameraView
from config.settings import (
    CALIBRATION_FRAMES,
    CALIBRATION_FACE_FRONT_MIN,
    CALIBRATION_FACE_BACK_MAX,
    CALIBRATION_TIMEOUT_FRAMES
)

logger = logging.getLogger(__name__)

class DynamicZoneManager:
    """
    Dynamically identifies the camera layout and view (Side, Front, Back)
    based on a stability calibration phase.
    """
    
    def __init__(self, stability_threshold_px: float = 30.0):
        self.stability_threshold_px = stability_threshold_px
        self.reset()
        
    def reset(self):
        """Resets the state, crucial for batch-run cross-video bleeding prevention."""
        self.is_calibrated = False
        self.layout_type = None  # 'SIDE_BY_SIDE' or 'TOP_BOTTOM'
        self.split_val = 0.0
        self.camera_view = CameraView.UNKNOWN
        
        self._consecutive_stable_frames = 0
        self._history = []  
        self.face_visibility_count = 0
        self._update_count = 0
        
    def _center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, person_boxes: List[Tuple[int, int, int, int]], face_detected_this_frame: bool = False) -> bool:
        """
        Feed person bounding boxes every frame. Returns True if calibrated.
        """
        if self.is_calibrated:
            return True
            
        self._update_count += 1
        
        # We need exactly 2 people to calibrate
        if len(person_boxes) < 2:
            if self._update_count == CALIBRATION_TIMEOUT_FRAMES:
                logger.warning(f"[ZoneManager] Timeout reached ({CALIBRATION_TIMEOUT_FRAMES} frames) without 2 pilots. Holding at UNKNOWN state.")
            return False
            
        # Get the two largest boxes
        sorted_boxes = sorted(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        top_2 = sorted_boxes[:2]
        
        c1 = self._center(top_2[0])
        a1 = (top_2[0][2] - top_2[0][0]) * (top_2[0][3] - top_2[0][1])
        c2 = self._center(top_2[1])
        a2 = (top_2[1][2] - top_2[1][0]) * (top_2[1][3] - top_2[1][1])
        
        # Sort centers by X just to keep them paired roughly
        if c1[0] < c2[0]:
            centers = [(c1, a1), (c2, a2)]
        else:
            centers = [(c2, a2), (c1, a1)]
        self._history.append(centers)
        
        if face_detected_this_frame:
            self.face_visibility_count += 1
            
        if len(self._history) >= CALIBRATION_FRAMES:
            self._finalize_calibration()
            
        return self.is_calibrated
        
    def _finalize_calibration(self):
        c1_list = [pair[0][0] for pair in self._history]
        c2_list = [pair[1][0] for pair in self._history]
        a1_list = [pair[0][1] for pair in self._history]
        a2_list = [pair[1][1] for pair in self._history]
        
        med_c1_x = np.median([c[0] for c in c1_list])
        med_c1_y = np.median([c[1] for c in c1_list])
        med_c2_x = np.median([c[0] for c in c2_list])
        med_c2_y = np.median([c[1] for c in c2_list])
        
        med_a1 = np.median(a1_list)
        med_a2 = np.median(a2_list)
        area_ratio = max(med_a1, med_a2) / max(1.0, min(med_a1, med_a2))
        
        dist_x = abs(med_c1_x - med_c2_x)
        dist_y = abs(med_c1_y - med_c2_y)
        
        logger.info(f"[ZoneManager] Face visibility during calibration: {self.face_visibility_count}/{CALIBRATION_FRAMES}, Area Ratio: {area_ratio:.2f}")
        
        if dist_x > dist_y:
            self.layout_type = 'SIDE_BY_SIDE'
            self.split_val = (med_c1_x + med_c2_x) / 2.0
            
            if area_ratio >= 2.0:
                self.camera_view = CameraView.SIDE
                print(f"\n[ZoneManager] Calibrated: SIDE_VIEW (Side-by-Side, split x={self.split_val:.1f}, area ratio={area_ratio:.2f})")
                logger.info(f"[ZoneManager] Calibrated: SIDE_VIEW (geometric detection)")
                self.is_calibrated = True
            elif self.face_visibility_count >= CALIBRATION_FACE_FRONT_MIN:
                self.camera_view = CameraView.FRONT
                print(f"\n[ZoneManager] Calibrated: FRONT_VIEW (Side-by-Side, split x={self.split_val:.1f})")
                logger.info(f"[ZoneManager] Calibrated: FRONT_VIEW")
                self.is_calibrated = True
            elif self.face_visibility_count <= CALIBRATION_FACE_BACK_MAX:
                self.camera_view = CameraView.BACK
                print(f"\n[ZoneManager] Calibrated: BACK_VIEW (Side-by-Side, split x={self.split_val:.1f})")
                logger.warning(f"[ZoneManager] Calibrated: BACK_VIEW. Drowsiness mathematically impossible from behind, disabling.")
                self.is_calibrated = True
            else:
                self.camera_view = CameraView.UNKNOWN
                print(f"\n[ZoneManager] UNCERTAIN: Face count {self.face_visibility_count} is in ambiguity band. Extending calibration.")
                logger.warning(f"[ZoneManager] UNCERTAIN: Face count {self.face_visibility_count} is in ambiguity band. Extending calibration.")
                # Extend calibration window by dropping the oldest half
                self._history = self._history[CALIBRATION_FRAMES // 2:]
                self.face_visibility_count = self.face_visibility_count // 2
                self.is_calibrated = False
        else:
            self.layout_type = 'TOP_BOTTOM'
            self.split_val = (med_c1_y + med_c2_y) / 2.0
            self.camera_view = CameraView.SIDE
            print(f"\n[ZoneManager] Calibrated: SIDE_VIEW (Top-Bottom, split y={self.split_val:.1f})")
            logger.info(f"[ZoneManager] Calibrated: SIDE_VIEW")
            self.is_calibrated = True
            
    def assign_pilots(self, person_boxes: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Assigns boxes to pid=1 and pid=2 based on the calibrated layout.
        Uses relative sorting for the two largest boxes to be robust against shifting.
        Falls back to absolute split_val if only 1 box is present.
        Blocks assignment if UNKNOWN.
        """
        if not self.is_calibrated or not person_boxes or self.camera_view == CameraView.UNKNOWN:
            return {}
            
        area = lambda b: (b[2]-b[0])*(b[3]-b[1])
        top_boxes = sorted(person_boxes, key=area, reverse=True)[:2]
        
        result = {}
        
        if len(top_boxes) == 2:
            if self.layout_type == 'SIDE_BY_SIDE':
                top_boxes.sort(key=lambda b: self._center(b)[0])
                result[1] = top_boxes[0]
                result[2] = top_boxes[1]
            else:
                top_boxes.sort(key=lambda b: self._center(b)[1])
                result[2] = top_boxes[0]
                result[1] = top_boxes[1]
        else:
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

