import numpy as np
import cv2
import matplotlib.pyplot as plt
from Yolo_detector import detector

class Kalmanfilter():
    def __init__(self,m,dt,std_a):
        self.m = m   # No.of Measurements ( x,y,a,h) 
        self.dt = dt 
        self.std_a = std_a # Standard deviation of acceleration

        self.A = np.eye(2*m,2*m) # State Transistion Matrix - A (2*m,2*m) 
        for i in range(m):
            self.A[i][m+i] = dt

        self.B = np.zeros((2*m,m)) # Control Matrix - B (2*m,m)
        for i in range(m):
            self.B[i][i] = (dt**2)/2
            self.B[i+m][i] = (dt)

        self.H = np.zeros((m,2*m)) # Transformation Matrix - H (m,2*m)
        for i in range(m):
            self.H[i][i] = 1

        self.Q = np.zeros((2*m,2*m)) # Process Covariance Matrix - q (2*m,2*m)
        for i in range(m):
            self.Q[i][i] = (dt**4)/4
            self.Q[i][i+m] = (dt**3)/2
            self.Q[i+m][i] = (dt**3)/2
            self.Q[i+m][i+m] = dt**2
        self.Q = self.Q*(std_a**2)

        self.R = np.zeros((m,m)) # Measurement Noise Covariance - R (m,m)
        for i in range(m):
            self.R[i][i] = (dt**4)/4
        self.R = self.R*(std_a**2)

        self.X = np.zeros((2*m,1))
        self.U = np.ones((m,1))
        self.P = np.eye(self.A.shape[1])


    def show(self):
        print(f'A :\n{self.A,self.A.shape}')
        print(f'B :\n{self.B,self.B.shape}')
        print(f'H :\n{self.H,self.H.shape}')
        print(f'Q :\n{self.Q,self.Q.shape}')
        print(f'R :\n{self.R,self.R.shape}')
        print(f'X :\n{self.X,self.X.shape}')
        print(f'U :\n{self.U,self.U.shape}')
        print(f'P :\n{self.P,self.P.shape}')
    
    def initialize(self,measurements):
        pos = measurements.reshape(4,1)
        vel = np.zeros_like(pos)
        self.X = np.r_[pos,vel]
        dia = []
        for i in range(2*self.m):
            if i<4:
                dia.append(((self.dtdt**2)*self.std_a)/2)
            else:
                dia.append(self.dt*self.std_a)
        self.P = np.diag(dia)
        return self.X,self.P

    def predict(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.U)
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        return self.X,self.P
    
    def update(self,z):
        s = np.dot(np.dot(self.H,self.P),self.H.T) + self.R
        self.K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(s))
        self.X = np.round(self.X + np.dot(self.K,(z - np.dot(self.H,self.X))))
        self.P = np.dot((np.eye(self.A.shape[1]) - np.dot(self.K,self.H)),self.P)
        return self.X,self.P
    

# result = detector()
# cls_name,m = result
# x,y,w,h = m[0].item(),m[1].item(),m[2].item(),m[3].item()
# z = np.array([x,y,w,h]).reshape(4,1).tolist()
# print(cls_name,z)

# kf = Kalmanfilter(4,0.1,1)
# predict = kf.predict()
# update = kf.update(z)
# predict1 = kf.predict()
# print(f"""
# predict (Before Updating) - {predict}
# update - {update}
# predict (After Updating) - {predict1}
# Actual Measurements - {z}
# """)

def bounding_box(m):
    x,y,w,h = m
    x1 = int((2*x-w)/2)
    x2 = int((2*x+w)/2)
    y1 = int((2*y-h)/2)
    y2 = int((2*y+h)/2)
    top_left = (x1,y1)
    bottom_right = (x2,y2)
    return top_left,bottom_right

# def main():
#     pred_start_point,pred_end_point= bounding_box(predict)
#     ori_start_point,ori_end_point = bounding_box(z)
# #     print(f"""
# # pred - {pred_start_point,pred_end_point}
# # orig - {ori_start_point,ori_end_point}
# # """)
#     image = cv2.imread('./sample_3.jpeg')
#     cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     cv2.rectangle(image,pred_start_point,pred_end_point,(0,0,0),20)
#     cv2.rectangle(image,ori_start_point,ori_end_point,(0,255,0),5)
#     cv2.imwrite('./result0.jpg',image)
#     # plt.imshow(image)
#     # cv2.imwrite('/home/mrgupta/mimic-_-/DRDO/kalman_filter_8d.py/predicted.jpg',image)

# if __name__ == '__main__':
#     main()


# def main():
#     videocap = cv2.VideoCapture(0)
#     kf = Kalmanfilter(4,0.1,1)
#     while True:
#         ret,frame = videocap.read() # Reading the frame
#         cls_name,m = detector(frame)
#         x,y,w,h = m[0].item(),m[1].item(),m[2].item(),m[3].item()
#         z = np.array([x,y,w,h]).reshape(4,1).tolist()
#         if len(z) > 0:
#             ori_start_point,ori_end_point = bounding_box(z)
#             cv2.rectangle(frame,ori_start_point,ori_end_point,(0,255,0),20) # Drawing Original bounding box
#             predict = kf.predict()
#             pred_start_point,pred_end_point = bounding_box(predict)
#             cv2.rectangle(frame,pred_start_point,pred_end_point,(0,0,0),5)
#             kf.update(z)
#         cv2.imshow('image',frame)




################################### Videooo   . . . .single object single classsss
# def main():
#     videocap = cv2.VideoCapture(0)
#     kf = Kalmanfilter(4,0.1,2)
#     while True:
#         ret,frame = videocap.read()
#         detect = detector(frame)
#         if detect != None:
#             cls_name,m = detect[0],detect[1]
#             x,y,w,h = m[0],m[1],m[2],m[3]
#             z =  np.array([x,y,w,h]).reshape(4,1).tolist()
#             ori_start_point,ori_end_point = bounding_box(z)
#             predict = kf.predict()
#             pred_start_point,pred_end_point = bounding_box(predict)
#             update = kf.update(z)
#             cv2.rectangle(frame,ori_start_point,ori_end_point,(0,255,0),15)
#             cv2.rectangle(frame,pred_start_point,pred_end_point,(0,0,0),5)
#         cv2.imshow('Video',frame)


#         if cv2.waitKey(2) & 0xFF == ord('q'):
#             videocap.release()
#             cv2.destroyAllWindows()
#             break
  


def main():
    videocap = cv2.VideoCapture(0)
    kf = Kalmanfilter(4,0.1,2)
    while True:
        ret,frame = videocap.read()
        detections = detector(frame)
        if detections !=None:
            for d in detections:
                ori_start,ori_end = bounding_box(d)
                z = np.array(d).reshape(4,1)
                kf.predict()
                p = kf.update(z)
                end_start,end_end = bounding_box(p.reshape(4))
                cv2.rectangle(frame,ori_start,ori_end,(0,255,0),15)
                cv2.rectangle(frame,end_start,end_end,(0,0,0),5)

        cv2.imshow('Video',frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            videocap.release()
            cv2.destroyAllWindows()
            break


                

# def main():
#     videocap = cv2.VideoCapture(0)
#     kf = Kalmanfilter(4,0.1,2)
#     while True:
#         ret,frame = videocap.read()
#         detect = detector(frame)
#         if detect != None:
#             cls,bb = detect[0],detect[1]
#             for b in bb:
#                 z = np.array([b[0],b[1],b[2],b[3]]).reshape(4,1).tolist()
#                 ori_start,ori_end = bounding_box(z)
#                 predict = kf.predict()
#                 pred_start,pred_end = bounding_box(predict)
#                 update = kf.update(z)
#                 cv2.rectangle(frame,ori_start,ori_end,(0,255,0),15)
#                 cv2.rectangle(frame,pred_start,pred_end,(0,0,0),5)
#         cv2.imshow('video',frame)

#         if cv2.waitKey(2) & 0xFF == ord('q'):
#             videocap.release()
#             cv2.destroyAllWindows()
#             break


if __name__ == '__main__':
    main()

