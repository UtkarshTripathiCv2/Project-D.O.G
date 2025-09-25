import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('DOG.pt')
CONFIDENCE_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.81
HIGH_ACCURACY_CLASSES = [
    'Healthy Wheat', 'apple black rot', 'apple healthy', 'apple scab',
    'bell pepper bacterial spot', 'bell pepper healthy', 'cedar apple rust',
    'cherry healthy', 'cherry powdery mildew', 'corn cerespora leaf spot',
    'corn common rust', 'corn healthy', 'grape black rot', 'grape esca',
    'grape healthy', 'grape leaf blight', 'cassava green mottle',
    'northern leaf blight', 'orange citrus greening', 'peach bacterial spot',
    'peach healthy', 'potato early blight', 'potato healthy', 'potato late blight',
    'rice brown spot', 'rice healthy', 'rice hispa', 'rice leaf blast',
    'spider mites two-spotted spider mite', 'squash powdery mildew',
    'strawberry healthy', 'strawberry leaf scorch', 'tomato bacterial spot',
    'tomato early blight', 'tomato late blight', 'tomato leaf healthy',
    'tomato leaf mould', 'Apple leaf', 'Corn rust leaf', 'Soyabean leaf',
    'Strawberry leaf', 'Tomato Septoria leaf spot', 'grape leaf black rot',
    'COW', 'pig'
]
np.random.seed(42)
colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(model.names))]

# --- Main Logic ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if success:
        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = box.conf[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name in HIGH_ACCURACY_CLASSES and confidence > CONFIDENCE_THRESHOLD:
                    color = colors[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    label = f'{class_name} {confidence:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    # Position for the background rectangle of the label
                    label_bg_x2 = x1 + text_width + 10
                    label_bg_y2 = y1 + text_height + baseline + 10
                    
                    # Position for the text itself
                    text_x = x1 + 5
                    text_y = y1 + text_height + 5

                    if confidence > HIGH_CONFIDENCE_THRESHOLD:

                        cv2.rectangle(frame, (x1, y1), (label_bg_x2, label_bg_y2), (255, 255, 255), -1)
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    else:

                        cv2.rectangle(frame, (x1, y1), (label_bg_x2, label_bg_y2), color, -1)
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Live Detection (Filtered)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
