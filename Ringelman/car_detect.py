import torch
import cv2
import numpy as np
from Ringelman.models.yolo3.yolo_layer import YoLoV3
from Ringelman.models.yolo3.config import cfg,HyperParameter_data as cfg_data
from Ringelman.models.yolo3.loss.pred_transform import PredTransform
from Ringelman.models.yolo3.utils import ResImgProcess

class CarDetect():
    def __init__(self,top,output_size):
        self.top = top
        self.model = YoLoV3(cfg).eval()
        self.model.load_state_dict(torch.load("./Base320_D1_FOCAL_yolov3_C3_E180.snap")["model"])
        self.model.cuda()
        self.pred_trans = PredTransform(cfg,output_size)
        self.pred_process = ResImgProcess(cfg,cfg_data)
        self.pred_res = None
        self.select_idx = 0
        self.calib_box_rel=[]

    def data_build(self,image,mean=None):
        image = cv2.resize(image,(320,320))
        image = image.astype(np.float32)
        if not mean is None:
            image = image - mean
        image = np.expand_dims(image,0)
        image = np.transpose(image,(0,3,1,2))
        return image

    def __call__(self, img):
        input_data = self.data_build(img,mean=(104, 117, 123))
        input_data_cuda = torch.from_numpy(input_data).float().cuda()
        logit = self.model(input_data_cuda)
        pred = self.pred_trans(logit)
        self.pred_res = pred
        return pred

    def draw_box(self,img):
        if len(self.pred_res[0])<=0 or len(self.pred_res[1])<=0:
            pass
        else:
            for box_idx,(box,label) in enumerate(zip(self.pred_res[0].detach().cpu().numpy().astype(np.int),
                                 self.pred_res[1].detach().cpu().numpy().astype(np.int))):
                w = box[2] - box[0]
                h = box[3] - box[1]
                if box_idx == self.select_idx:
                    color = (0,0,255)
                    if self.top.rbtn_var.get()==1 and len(self.calib_box_rel)<=0 and len(self.top.show_canvas.calib_box)>=4:
                        calib_box = np.array(self.top.show_canvas.calib_box)
                        calib_box = (calib_box - (box[0],box[1]))/(w,h)
                        self.calib_box_rel = calib_box.tolist()
                        self.top.show_canvas.points_clear()
                        print("")
                else:
                    color = (0, 255, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)
                if len(self.calib_box_rel)>=4:
                    calib_box = []
                    for point_idx,point in enumerate(self.calib_box_rel):  # 按照lt,lb,rb,rt的顺序排列
                        calib_box.append((box[0]+int(point[0]*w),box[1]+int(point[1]*h)))
                    for point_idx in range(4):
                        cv2.line(img,calib_box[point_idx],calib_box[point_idx-1],color)
                cv2.putText(img, "car", (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)