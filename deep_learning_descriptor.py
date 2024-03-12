import numpy as np
import cv2 
import tensorflow as tf
from tensorflow.keras.applications import vgg16

class DeepFeatures():
    def __init__(self,image,bb):
        img = cv2.imread(image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.image = img ## Matrix image
        self.bb = np.array(bb).reshape(4).tolist() ## List of Bounding Box Containing (Xc,Yc,a,h)
        self.model = vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
    
    def crop_image(self):
        x,y,a,h = self.bb
        w = a*h
        x1 = int((2*x-w)/2)
        x2 = int((2*x+w)/2)
        y1 = int((2*y-h)/2)
        y2 = int((2*y+h)/2)
        crop_img = self.image[y1:y2,x1:x2]
        return crop_img ## Croping the image with the bounding box measurements
    
    def preprocess_image(self):
        crop_img = self.crop_image()
        prepro_img = cv2.resize(crop_img,(224,224))
        prepro_img = vgg16.preprocess_input(prepro_img)
        prepro_img = np.expand_dims(prepro_img,axis=0)
        return prepro_img
    
    def model(self):
        prepro_img = self.preprocess_image()
        fea = self.model.predict(prepro_img)
        fea = fea.flatten()
        return fea
    

