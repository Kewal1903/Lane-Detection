import cv2 as cv
import numpy as np


video_path = "C:\\Users\\Kewal\\Desktop\\3059073-hd_1920_1080_24fps.mp4"
cap = cv.VideoCapture(video_path)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

prev_frame = None
color = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv.resize(frame,(640, 480),interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(7, 7),0)
    canny = cv.Canny(blur,50,150)
    cv.imshow("Canny",canny)

    if prev_frame is None:
        corners = cv.goodFeaturesToTrack(canny,maxCorners=100,qualityLevel=0.3,minDistance=4)
        p0 = np.float32(corners)
        prev_frame = canny
        continue

    p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame,canny,p0,None,**lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    lines = cv.HoughLinesP(canny,1, np.pi / 180,threshold=50,minLineLength=40,maxLineGap=0)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(resized,(x1, y1),(x2, y2),(0, 255, 0),2)

    cv.imshow("Lane Detection",resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = canny.copy()

cap.release()
cv.destroyAllWindows()
