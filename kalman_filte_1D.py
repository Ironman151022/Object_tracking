import numpy as np
import matplotlib.pyplot as plt

class kalmanfilter(object):
    def __init__(self,dt,u,std_acc,std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.meas = std_meas
        self.A = np.matrix([[1,self.dt],
                            [0,1]])
        self.B = np.matrix([[0.5*(self.dt)**2],
                            [self.dt]])
        self.H = np.matrix([1,0])
        self.q = np.matrix([[((self.dt)**4)/4,((self.dt)**3)/2],
                            [((self.dt)**3)/2,((self.dt)**2)]])*self.std_acc**2
        self.r = std_meas**2
        self.p = np.eye(self.A.shape[1])
        self.x = np.matrix([[0],[0]])

    def predict(self): # Time update equations
        self.x = np.dot(self.A,self.x) + np.dot(self.B, self.u) # predicted state estimate
        self.p = np.dot(np.dot(self.A,self.p),self.A.T) + self.q # predicted Error Covariance
        return self.x

    def update(self,z):
        s = np.dot(np.dot(self.H,self.p),self.H.T) + self.r
        k = np.dot(np.dot(self.p,self.H),np.linalg.inv(s))
        self.x = np.round(self.x + np.dot(k,(z - np.dot(self.H,self.x))))
        self.p = (np.eye(self.H.shape[1])-np.dot(k,self.H))*self.p

                                                                                                
def main():
    dt = 0.1
    t = np.arange(0,100,dt)
    real_track = 0.1*((t**2)-t)
    u = 2
    std_acc = 0.25
    std_meas = 1.2
    kf = kalmanfilter(dt,u,std_acc,std_meas)
    predictions = []
    measurements = []
    for x in real_track:
        z = kf.H*x + np.random.normal(0,50)
        measurements.append(z.item(0))
        predictions.append(kf.predict()[0])
        kf.update(z.item(0))

    fig = plt.figure()
    fig.suptitle("Kalman Filter 1-D tracking")
    fig.plot(t,measurements,label='Measurements',color='b')
    fig.plot(t,np.squeeze(predictions),label='predictions',color='r')
    fig.plt(t,np.array(real_track),label='real_track',color='y')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()

                           
