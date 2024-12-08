import cv2 as cv
import numpy as np
from ultralytics import YOLO
import mss
import threading

class VisionCapture:
    def __init__(self, model_path):
        # Initialise the YOLO model for object detection
        self.model = YOLO(model_path)
        self.frame = None
        self.detected_frame = None
        self.frame_lock = threading.Lock()
        self.best_score_width = 0

        # Start screen capture in a separate thread
        self.capture_thread = threading.Thread(target=self.screen_capture)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start object detection in a separate thread
        self.detection_thread = threading.Thread(target=self.object_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def screen_capture(self):
        # Screen is captured continuously using MSS
        sct = mss.mss()
        while True:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            with self.frame_lock:
                self.frame = np.array(screenshot)
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGRA2BGR)

    def object_detection(self):
        while True:
            if self.frame is not None:
                with self.frame_lock:
                    input_frame = self.frame.copy()
                
                # Perform object detection
                results = self.model(input_frame, show=False)
                
                max_score_width = 0
                detected_boxes = []

                for result in results:
                    for box in result.boxes:
                        conf = box.conf.cpu().numpy()[0]
                        
                        if conf < 0.50:
                            continue  

                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        cls = int(box.cls.cpu().numpy()[0])
                        label = self.model.names[cls]

                        # Calculate width of bounding box
                        width = x2 - x1

                        # Check for the score class and select the bounding box with the largest width to avoid multiple bounding boxes
                        if label == 'score' and width > max_score_width:
                            max_score_width = width
                        
                        # Add detection to the list
                        detected_boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))

                # Store the best score width to be used later as a reward
                self.best_score_width = max_score_width

                with self.frame_lock:
                    self.detected_frame = input_frame.copy()
                    self.detected_boxes = detected_boxes  # Store the detected boxes
    
    # Method to retrieve the latest detected frame and score width
    def get_detected_frame_and_score_width(self):
        with self.frame_lock:
            if self.detected_frame is None:
                return None, []
            return self.detected_frame, self.detected_boxes  # Return the list of detected boxes
