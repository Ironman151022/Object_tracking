import numpy as np
import cv2

class KalmanFilter():
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

    def predict(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.U)
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        return self.X,self.P
    
    def update(self,z):
        s = np.dot(np.dot(self.H,self.P),self.H.T) + self.R
        self.K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(s))
        self.X = np.round(self.X + np.dot(self.K,(z - np.dot(self.H,self.X))))
        self.P = np.dot((np.eye(self.A.shape[1]) - np.dot(self.K,self.H)),self.P)
        return self.X,self.P  # measurement matrix and covariance matrix
    
    
