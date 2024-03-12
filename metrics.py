from final_kalman import KalmanFilter
from deep_learning_descriptor import DeepFeatures
import numpy as np
from scipy.stats import chi2

class Metrics(KalmanFilter,DeepFeatures):
    def __init__(self,image,actual_bb,prediction_state):
        self.fea = DeepFeatures
        self.image = image
        self.actual_bb = actual_bb
        self.pre_mean = prediction_state[0]
        self.pre_cov = prediction_state[1]
        self.maha_threshold = chi2.ppf(0.95,2)
        self.lamda = 0

    def mahalanobis_distance(self):
        z = np.array(self.actual_bb[:2]).reshape(2,1)
        p = self.pre_mean[:2]
        q = self.pre_cov[:2,:2]
        distance = np.linalg.multi_dot([(p-z).T,q,(p-z)])
        return distance
    
    def cosine_distance(self):
        actual_fea = self.fea(self.image,self.actual_bb)
        pre_fea = self.fea(self.image,self.pre_mean[:4])
        
        dot = np.dot(actual_fea,pre_fea)
        norm_a = np.linalg.norm(actual_fea)
        norm_b = np.linalg.norm(pre_fea)
        distance = np.divide(dot,np.multiply(norm_a,norm_b))
        return distance
    





