import torch
from ultralytics import YOLO
import cv2

# Training code


# model = YOLO("yolov8n.pt")  

# model.train(
#    data="E:\Khaled\Khaled_Ayman_Perception_Tasks\Task2_WS\Task2\cars.yaml",     
#    epochs=50,            
#    imgsz=512,            
#    batch=4,              
#    workers=0,            
#    device=0              
# )


model = YOLO(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\Task2_WS\runs\detect\train2\weights\best.pt')
img = cv2.imread(r'C:\Users\Khaled\Desktop\OPENCV\imgs\ferrari.jpg')

model.predict(img, show = True , save = True , save_dir=r"E:\Khaled\Khaled_Ayman_Perception_Tasks\Task2_WS\results" , show_conf=False)


cv2.waitKey(0)

