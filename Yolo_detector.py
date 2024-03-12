from ultralytics import YOLO

model = YOLO('yolov8m.pt') # Loading the model 

def detector(s=0):
    results = model.predict(source=s,classes = 79)
    bb = results[0].boxes.xywh
    if len(bb)>0:
        return bb[0].tolist()
    else:
        return None

    
# print(detector('/home/mrgupta/mimic-_-/DRDO/test_images/sample_3.jpeg'))

    # cls = results[0].boxes.cls[0].item()
    # cls_name = results[0].names[cls]
    # m = results[0].boxes.xywh[0]


