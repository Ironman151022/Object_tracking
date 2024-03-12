import cv2
from yolov8 import detector
from final_kalman import KalmanFilter


def main():
    videocap = cv2.VideoCapture(0)
    k = KalmanFilter(4,0.1,10)
    while True:
        ret,frame = videocap.read()
        bb = detector(frame)
        if bb != None:
            pre_mean,pre_cov = k.predict(bb)
            up_mean,up_cov = k.update(bb)
            d = k.gatting_distance(bb)
            print(d)

        # cv2.imshow('image',frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            videocap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()


    



