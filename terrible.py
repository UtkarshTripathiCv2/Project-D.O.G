import cv2
from ultralytics import YOLO


model = YOLO('best.pt')


CONFIDENCE_THRESHOLD = 0.6

HIGH_ACCURACY_CLASSES = [
    'Healthy Wheat',
    'apple black rot',
    'apple healthy',
    'apple scab',
    'bell pepper bacterial spot',
    'bell pepper healthy',
    'cedar apple rust',
    'cherry healthy',
    'cherry powdery mildew',
    'corn cerespora leaf spot',
    'corn common rust',
    'corn healthy',
    'grape black rot',
    'grape esca',
    'grape healthy',
    'grape leaf blight',
    'cassava green mottle',
    'northern leaf blight',
    'orange citrus greening',
    'peach bacterial spot',
    'peach healthy',
    'potato early blight',
    'potato healthy',
    'potato late blight',
    'rice brown spot',
    'rice healthy',
    'rice hispa',
    'rice leaf blast',
    'spider mites two-spotted spider mite',
    'squash powdery mildew',
    'strawberry healthy',
    'strawberry leaf scorch',
    'tomato bacterial spot',
    'tomato early blight',
    'tomato late blight',
    'tomato leaf healthy',
    'tomato leaf mould',
    'Apple leaf',
    'Corn rust leaf',
    'Soyabean leaf',
    'Strawberry leaf',
    'Tomato Septoria leaf spot',
    'grape leaf black rot',
    'COW',
    'pig'
]


# --- Main Logic ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if success:
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = box.conf[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

               
                if class_name in HIGH_ACCURACY_CLASSES and confidence > CONFIDENCE_THRESHOLD:
                    
                 
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box for high confidence

                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Detection (Filtered)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
