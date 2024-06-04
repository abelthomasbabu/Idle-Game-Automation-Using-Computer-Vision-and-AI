import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pyautogui
import keyboard

# Loading the custom trained YOLO model only focusing on cookie clicker
# Switch between models for comparison
model = YOLO('Models\CookieOnly_revised_model_40_Epochs.pt')

# The clicker can be toggled on/off using Shift+s
clicking = False

def toggle_clicking():
    global clicking
    clicking = not clicking

# Set up the key combination for toggling
keyboard.add_hotkey('shift+s', toggle_clicking)

cv.namedWindow('GameObjectDetection', cv.WINDOW_NORMAL)

while True:
    # Capture the screen
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to a NumPy array
    frame = np.array(screenshot)

    # Convert RGB to BGR (OpenCV uses BGR)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if clicking:

        results = model(frame, show=False)

        primary_button_detected = False

        # Draw bounding boxes and click on primary_button
        for result in results:
            for box in result.boxes:
                conf = box.conf.cpu().numpy()[0]
                
                if conf < 0.2:
                    continue  

                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                # Test Clicker
                # Annotated class names in roboflow
                # Add class names here to be clicked
                if model.names[cls] in ['primary_btn']:
                    # Calculate the center of the bounding box to mouse click
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # pyautogui.click(center_x, center_y)

                # Draw bounding boxes for visualisation
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv.putText(frame, f'{model.names[cls]} {conf:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display the frame
    cv.imshow('GameObjectDetection', frame)

    # Press q to exit window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

