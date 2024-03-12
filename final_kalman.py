import numpy as np

class KalmanFilter():
    def __init__(self,m,dt,std_a):
        self.m = m
        self.dt = dt
        self.std_a = std_a

        self.A = np.eye(m*2)  ### A
        for i in range(m):
            self.A[i][m+i] = dt
        
        self.H = np.eye(m,2*m)  ### H

        self.X = []
        self.P = []

        # pos =  np.array(measurements).reshape(4,1)
        # vel = np.zeros_like(pos)
        # self.X = np.r_[pos,vel] ### initial X - mean (8x1)

        # std_cov = [
        #     (self.dt**4)/4 * measurements[3],
        #     (self.dt**4)/4 * measurements[3],
        #     (self.dt**4)/4 * measurements[3],
        #     1e-2,
        #     (self.dt**2) * measurements[3],
        #     (self.dt**2) * measurements[3],
        #     1e-5,
        #     (self.dt**2) * measurements[3],
        # ] 

        # self.P = np.diag(std_cov) * (self.std_a**2) ### Initial P - Covarince (8x8)

    def predict(self,measurements):

        if len(self.X) == 0 :
            pos =  np.array(measurements).reshape(4,1)
            vel = np.zeros_like(pos)
            self.X = np.r_[pos,vel] ### initial X - mean (8x1)

            std_cov = [
                (self.dt**4)/4 * measurements[3],
                (self.dt**4)/4 * measurements[3],
                (self.dt**4)/4 * measurements[3],
                1e-2,
                (self.dt**2) * measurements[3],
                (self.dt**2) * measurements[3],
                1e-5,
                (self.dt**2) * measurements[3],
            ] 

            self.P = np.diag(std_cov) * (self.std_a**2) ### Initial P - Covarince (8x8)

        std_q = [
            (self.dt**4)/4 * self.X[3].item(),
            (self.dt**4)/4 * self.X[3].item(),
            (self.dt**4)/4 * self.X[3].item(),
            1e-2,
            (self.dt**2) * self.X[3].item(),
            (self.dt**2) * self.X[3].item(),
            1e-5,
            (self.dt**2) * self.X[3].item(),
        ] 
        Q = np.diag(std_q) * self.std_a**2
        self.X = np.dot(self.A,self.X)
        self.P = np.linalg.multi_dot([self.A,self.P,self.A.T]) + Q

        return self.X,self.P
    
    def update(self,measurements):
        std_r = [
            (self.dt**4)/4 * self.X[3].item(),
            (self.dt**4)/4 * self.X[3].item(),
            1e-2,
            (self.dt**2) * self.X[3].item()] 
        
        R = np.diag(std_r) * (self.std_a**2)
    
        s = np.linalg.multi_dot([self.H,self.P,self.H.T]) + R
        k = np.linalg.multi_dot([self.P,self.H.T,np.linalg.inv(s)])
        innovation = (np.array(measurements).reshape(4,1)-np.dot(self.H,self.X))
        self.X = self.X + np.dot(k,innovation)
        self.P = np.dot(np.eye(self.X.shape[0]) - np.dot(k,self.H),self.P) 
        return self.X,self.P
    
    def gatting_distance(self,measurements):
        innovation = np.array(measurements).reshape(4,1)[:2] - self.X[:2]
        cov = self.P[:2,:2]
        d = np.linalg.multi_dot([innovation.T,np.linalg.inv(cov),innovation])
        return d
    