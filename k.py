import numpy as np

class k():
    def __init__(self,m,dt,std_a):
        self.m = m
        self.dt = dt
        self.std_a = std_a

        self.A = np.eye(m*2)  ### A
        for i in range(m):
            self.A[i][m+i] = dt
        
        self.H = np.eye(m,2*m)  ### H

    def intialize(self,measurements): ## measurements --  An array of 1,4 (x,y,a,h)
        pos =  np.array(measurements).reshape(4,1)
        vel = np.zeros_like(pos)
        mean = np.r_[pos,vel]

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

        cov = np.diag(std_cov) * (self.std_a**2)
        return mean,cov

    def predict(self,mean,cov): ## Predicting the mean, covariance
        std_q = [
            (self.dt**4)/4 * mean[3].item(),
            (self.dt**4)/4 * mean[3].item(),
            (self.dt**4)/4 * mean[3].item(),
            1e-2,
            (self.dt**2) * mean[3].item(),
            (self.dt**2) * mean[3].item(),
            1e-5,
            (self.dt**2) * mean[3].item(),
        ] 
        Q = np.diag(std_q) * self.std_a**2
        mean = np.dot(self.A,mean)
        cov = np.linalg.multi_dot([self.A,cov,self.A.T]) + Q

        return mean,cov
    
    # def project(self,mean,cov): 
    #     mean = np.dot(self.H,mean)
    #     std_r = [
    #         (self.dt**4)/4 * mean[3],
    #         (self.dt**4)/4 * mean[3],
    #         (self.dt**4)/4 * mean[3],
    #         1e-2,
    #         (self.dt**2) * mean[3]] * (self.std_a**2)
    #     R = np.diag(std_r)
    #     cov = np.linalg.multi_dot(self.H,cov,self.H.T) + R
    #     return mean,cov
    
    def update(self,mean,cov,measurements):
        std_r = [
            (self.dt**4)/4 * mean[3].item(),
            (self.dt**4)/4 * mean[3].item(),
            1e-2,
            (self.dt**2) * mean[3].item()] 
        
        R = np.diag(std_r) * (self.std_a**2)
    
        s = np.linalg.multi_dot([self.H,cov,self.H.T]) + R
        k = np.linalg.multi_dot([cov,self.H.T,np.linalg.inv(s)])
        innovation = (np.array(measurements).reshape(4,1)-np.dot(self.H,mean))
        mean = mean + np.dot(k,innovation)
        cov = np.dot(np.eye(mean.shape[0]) - np.dot(k,self.H),cov) 
        return mean,cov
    
    def gatting_distance(self,measurements,mean,cov):
        innovation = np.array(measurements).reshape(4,1)[:2] - mean[:2]
        cov = cov[:2,:2]
        d = np.linalg.multi_dot([innovation.T,np.linalg.inv(cov),innovation])
        return d
    

        