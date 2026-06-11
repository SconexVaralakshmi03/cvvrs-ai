import cv2
import os
import sys

# Ensure cvvrs-ai is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector.gadget_detector import YoloObjectDetector
from detector.seat_absence_detector import SeatAbsenceDetector
from detector.head_drop_detector import HeadDroopDetector
RAW_FRAME_SKIP = 3
GADGET_EVERY = 6
ABSENCE_EVERY = 4
DROOP_EVERY = 15

def main():
    source_vid = "test_videos/drow.mp4"
    out_vid = "test_videos/drow_debug.mp4"
    
    if not os.path.exists(source_vid):
        print(f"File not found: {os.path.abspath(source_vid)}")
        return

    cap = cv2.VideoCapture(source_vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    yolo = YoloObjectDetector()
    absence = SeatAbsenceDetector()
    droop = HeadDroopDetector()
    
    frame_no = 0
    processed_no = 0
    absence_pilot_boxes = []

    print(f"Processing {source_vid} for debugging metrics...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_no += 1
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if frame_no % RAW_FRAME_SKIP != 0:
            continue
            
        processed_no += 1
        annotated = frame.copy()
        
        # YOLO (run every GADGET_EVERY)
        run_yolo = processed_no % GADGET_EVERY == 0
        if run_yolo:
            results, _ = yolo.process(frame, round(video_time, 3))
            
            # Draw YOLO info directly from the hits before filtering
            for hit in yolo.last_object_hits:
                if hit.class_name == "cell phone":
                    x, y, x2, y2 = hit.bbox
                    w = x2 - x
                    h = y2 - y
                    aspect = round(w / h if h > 0 else 0, 2)
                    area = w * h
                    text = f"Phone AR={aspect} Conf={hit.confidence:.2f}"
                    cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated, text, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            absence_pilot_boxes = [(r.pilot_id, r.bbox) for r in results]

        # Seat Absence
        run_absence = processed_no % ABSENCE_EVERY == 0
        if run_absence:
            absence_res, _ = absence.process(absence_pilot_boxes, video_time, width, height)
            split_y = int(height * 0.57) # GREEN_LINE_RATIO
            cv2.line(annotated, (0, split_y), (width, split_y), (0, 255, 0), 2)
            for r in absence_res:
                if r.tracking_bbox:
                    bx1, by1, bx2, by2 = r.tracking_bbox
                    cy = (by1 + by2) / 2
                    cv2.circle(annotated, (int((bx1+bx2)/2), int(cy)), 5, (255, 0, 0), -1)
                    cv2.putText(annotated, f"P{r.pilot_id} cy={cy:.1f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Droop
        run_droop = processed_no % DROOP_EVERY == 0
        if run_droop:
            droop_res, _ = droop.process(frame, video_time, yolo.last_frame_detections)
        
        writer.write(annotated)
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} raw frames...")

    cap.release()
    writer.release()
    print(f"Finished. Debug video saved to {out_vid}")

if __name__ == '__main__':
    main()
