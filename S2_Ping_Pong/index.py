import cv2 
import numpy as np
import imutils

cap = cv2.VideoCapture('/Users/hientruongthmh/Documents/AI_VNUK/S2_Ping_Pong/pingpong.mp4')

while(True):
    _, frame = cap.read() # lưu biến video

    blur = cv2.GaussianBlur(frame, (11,11), 0) #lọc màu blur,
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # Đổi không gian màu BGR (Blue, Green, Red) sang không gian HSV

    lower = np.array([15, 130, 30])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper) # Mặt nạ không gian màu
    mask = cv2.erode(mask, None, iterations = 2) # Lọc nhiễu những điểm "Độc lập" == Đốm
    mask = cv2.dilate(mask, None, iterations = 2) # Hoàn chỉnh vật thể, tránh trường hợp do lọc nhiễu nên 1 vật thể bị tách ra

    #Vẽ đường bao quanh banh
    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tìm, Quét lấy banh
    ball_cnts = imutils.grab_contours(ball_cnts) # Đếm banh

    # Sau khi vẽ xong thì tìm trái banh: Có thể hiểu sau khi vẽ một đường tròn lớn bao quanh trái banh thì ta tìm trái banh là một đường tròn
    # nhỏ trong đường tròn lớn ấy
    if(len(ball_cnts) > 0) :
        c = max (ball_cnts, key = cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
    
        if radius > 0: 
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 2  )
 
    cv2.imshow('frame',frame) # tạo khung hình

    if cv2.waitKey(50) & 0xFF == ord('q'): # mở video
        break