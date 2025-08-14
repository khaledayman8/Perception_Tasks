from ultralytics import YOLO
import torch
import cv2
import cv2
import numpy as np

# Training code

# model = YOLO('yolov8n-seg.pt')
# model.train(
#     data=r'E:/Khaled/Khaled_Ayman_Perception_Tasks/BonusTask_WS/BonusTask/lanes.yaml',
#     epochs=50,
#     imgsz=512,
#     batch=4,
#     device=0
# )

video = cv2.VideoCapture(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\BonusTask_WS\highway__video.mp4')
detected_video = cv2.VideoWriter(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\BonusTask_WS\results from the trained model\detected__video.mp4' , cv2.VideoWriter_fourcc(*'mp4v') , 30,
    (1280, 720) )
model = YOLO(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\BonusTask_WS\runs\segment\train\weights\best.pt')

while True:
    ret , frame = video.read()

    if ret:
        frame = cv2.resize(frame , (1280 , 720))

        results = model(frame , show_boxes = False )

        cv2.imshow('video' , results[0].plot(boxes=False))
        detected_video.write(results[0].plot(boxes=False))
        if cv2.waitKey(1) >= 0:
            break
    else:
        break

video.release()
detected_video.release()
cv2.destroyAllWindows()
