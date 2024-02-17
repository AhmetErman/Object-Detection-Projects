from sort import *
from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture('../Videos/stairs2.mp4')  # Video
model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

count_down = []
count_up = []
past_position = {}

while True:
    ret, frame = cap.read()
    # img_region = cv2.bitwise_and(frame, mask)
    # results = model(img_region, stream=True)
    results = model(frame, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2 - x1, y2 - y1
            # cvzone.cornerRect(frame, bbox, l=8)

            # Confidence
            confidence = math.ceil(box.conf[0] * 1000) / 10

            # Class Name
            cls = int(box.cls[0])
            CurrentClass = classNames[cls]

            if CurrentClass == "person" and confidence > 30:
                current_array = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_array))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=0.7, thickness=2, offset=10)

        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(frame, (cx, cy), 3, (255, 255, 30), cv2.FILLED)

        if past_position.__contains__(id):
            if cy < past_position.get(id) and count_up.count(id) == 0:
                if count_down.count(id) > 0:
                    count_down.remove(id)
                count_up.append(id)
            elif cy > past_position.get(id) and count_down.count(id) == 0:
                if count_up.count(id) > 0:
                    count_up.remove(id)
                count_down.append(id)

        past_position.update({id: cy})

    cvzone.putTextRect(frame, f' Going UP: {len(count_up)}', (50, 50),scale=1,thickness=2, colorR=(255, 0, 0))
    cvzone.putTextRect(frame, f' Going DOWN: {len(count_down)}', (50, 100),scale=1,thickness=2, colorR=(0, 0, 255))

    cv2.imshow('Image', frame)
    cv2.waitKey(1)
