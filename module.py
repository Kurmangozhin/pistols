import numpy as  np
import cv2, os, random, colorsys



class Recognition(object):
    def __init__(self, path_cls:str,weghts:str, cfg:str):
        self.class_labels = self.read_classes(path_cls)
        self.class_colors = self.colors()
        self.yolo_model = cv2.dnn.readNet(weghts,cfg)
        self.model = cv2.dnn_DetectionModel(self.yolo_model)
        self.model.setInputParams(size = (416, 416), scale=1/255, swapRB=True)
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
    
   
    def read_classes(self, path:str):
        with open(f'{path}', 'r') as f:
             class_labels = f.readlines()
        class_labels = [cls.strip() for cls in class_labels]
        return class_labels
    
    
    def colors(self):
        hsv_tuples = [(x / len(self.class_labels), 1., 1.) for x in range(len(self.class_labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        class_colors = list(map(lambda x:(int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
        np.random.seed(43)
        np.random.shuffle(colors)
        np.random.seed(None)
        class_colors = np.tile(class_colors,(16,1))
        return class_colors   
    
   
    
    def detection(self, img:str,video = False):
        if video:
            img_to_detect = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_to_detect = cv2.imread(img, cv2.IMREAD_COLOR)
            img_to_detect = cv2.cvtColor(img_to_detect, cv2.COLOR_BGR2RGB)
        classes, scores, boxes = self.model.detect(img_to_detect, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        for (classid, score, box) in zip(classes, scores, boxes):
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height
            predicted_class_label = "{}: {:.2f}%".format(self.class_labels[classid[0]], score[0]*100)
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt, box_width, box_height), (255,255,0),2)
            cv2.putText(img_to_detect, predicted_class_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img_to_detect
    
    
    def detection_video(self, path:str, output_path:str,fps = 25):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        frame_height, frame_width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break    
            output  = self.detection(frame, video = True)
            output  = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            out.write(output)
        out.release()



if __name__ == '__main__':
    opt = {'path_cls':'classes.txt','weghts':'weights/pistol.weights','cfg':'weights/pistol.cfg'}
    cls = Recognition(**opt)
    cls.detection_video('input/12.mp4', 'output/1.webm')       





