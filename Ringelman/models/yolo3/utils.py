import cv2
import numpy as np
class ResImgProcess():
    def __init__(self,model_cfg,data_cfg):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

    def process(self, pred, img):
        checked = False
        if len(pred[0])<=0 or len(pred[1])<=0:
            return None
        for box,label in zip(pred[0].detach().cpu().numpy().astype(np.int),
                             pred[1].detach().cpu().numpy().astype(np.int)):
            if label in self.model_cfg["main_clses"]:
                if not checked:
                    checked = True
                if self.data_cfg["draw_box"]:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
                    cv2.putText(img, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            elif label in self.model_cfg["other_clses"]:
                if self.data_cfg["draw_box"]:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255))
                    cv2.putText(img, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        return checked