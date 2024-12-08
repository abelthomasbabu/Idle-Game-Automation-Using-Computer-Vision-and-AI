import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import mss
import threading

# Load the custom trained YOLO model
model = YOLO('vision_models/FinalModel198Epochs.pt')

window_name = 'GameObjectDetection'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

frame = None
detected_frame = None
frame_lock = threading.Lock()
fps = 0
best_score_width = 0

def screen_capture():
    global frame, fps
    sct = mss.mss()
    while True:
        start_time = time.time()
        
        monitor = sct.monitors[1] # The monitor number to be captured
        screenshot = sct.grab(monitor)
        with frame_lock:
            frame = np.array(screenshot)
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        fps = 1 / (time.time() - start_time)

def object_detection():
    global detected_frame, best_score_width
    while True:
        if frame is not None:
            with frame_lock:
                input_frame = frame.copy()
            
            # Perform object detection
            results = model(input_frame, show=False)
            
            max_score_width = 0
            best_score_box = None

            for result in results:
                for box in result.boxes:
                    conf = box.conf.cpu().numpy()[0]
                    
                    # Confidence kept low for testing purposes
                    if conf < 0.1:
                        continue  

                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    cls = int(box.cls.cpu().numpy()[0])
                    label = model.names[cls]

                    # Calculate width of bounding box
                    width = x2 - x1

                    # Check for the score class and select the bounding box with the largest width to avoid multiple bounding boxes
                    if label == 'score' and width > max_score_width:
                        max_score_width = width
                        best_score_box = (int(x1), int(y1), int(x2), int(y2), label, conf)

                    # Draw bounding boxes for other labels
                    elif label != 'score':
                        cv.rectangle(input_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv.putText(input_frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Draw only the best score bounding box
            if best_score_box:
                x1, y1, x2, y2, label, conf = best_score_box
                cv.rectangle(input_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv.putText(input_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
            with frame_lock:
                detected_frame = input_frame.copy()

# Start the screen capture thread
capture_thread = threading.Thread(target=screen_capture)
capture_thread.daemon = True
capture_thread.start()

# Start the object detection thread
detection_thread = threading.Thread(target=object_detection)
detection_thread.daemon = True
detection_thread.start()

while True:
    # Display the latest detected frame
    with frame_lock:
        if detected_frame is not None:
            # To display FPS on the frame
            cv.putText(detected_frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow(window_name, detected_frame)

    # Press 'q' to exit the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
