import cv2
import numpy as np

video = cv2.VideoCapture(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\Task1\data\Car.mp4')

fps = int(video.get(cv2.CAP_PROP_FPS))
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = cv2.VideoWriter(r'E:\Khaled\Khaled_Ayman_Perception_Tasks\Task1\result\output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(width, height))
while True:
    ret , frame = video.read()
    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5,5), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours_yellow:
            if cv2.contourArea(cnt) > 500: 
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)   

        for cnt in contours_blue:
            if cv2.contourArea(cnt) > 500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)  

        cv2.imshow('output_video' , frame)
        output_video.write(frame)

        if cv2.waitKey(1) >= 0:
            break

    else:
        break

video.release()
output_video.release()
cv2.destroyAllWindows()