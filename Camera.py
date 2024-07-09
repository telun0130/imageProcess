import cv2 as cv
import time
from detection import preprocess

center_width = 240
center_height = 720

def capture_frame():
    cap = cv.VideoCapture(0)  # 0表示第一個攝影鏡頭
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    desired_width = 1280
    desired_height = 720
    cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        center_x = (frame_width - center_width) // 2
        center_y = (frame_height - center_height) // 2

        try:
            result = preprocess(frame, Ct_x=center_x, Ct_y=center_y)
        except Exception as e:
            print(e)

        print('裂縫數：%s' % str(result[1]))
        cv.imshow('Frame', result[0])
        if cv.waitKey(1) & 0xFF == ord('q'):  # 'q'退出
            break
        time.sleep(1)  # 延遲以匹配禎速

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    capture_frame()