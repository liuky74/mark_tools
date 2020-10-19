from UI import Window
import cv2
import numpy as np
import data_io
from configs import ALL_CLS
cls_maps={}
for idx,cls in enumerate(ALL_CLS):
    cls_maps[idx] = cls

class DurationDataMarkWindow():
    def __init__(self, window_name):
        self.window_name = window_name
        self._show_img = None  # np.zeros((720,1280,3),dtype=np.uint8)
        # 高亮框的idx
        self.box_select_idx = 0
        # label,只会手动清除
        self.labels = []
        # 画框过程中保存临时框的坐标
        self.left_top_point = None
        self.right_bottom_point = None
        # 画框的框类别
        self.mark_cls = None
        cv2.setMouseCallback(self.window_name, self.draw_call_back)

    def clear_data(self):
        self.labels.clear()
        self.left_top_point=None
        self.right_bottom_point=None
        self.box_select_idx = 0

    def del_box(self):
        if (self.labels is None) or (len(self.labels))==0:
            return None
        for box_idx, box in enumerate(self.labels):
            if box_idx == self.box_select_idx:
                self.labels.pop(box_idx)
                if self.box_select_idx>0:
                    self.box_select_idx-=1

    def key_process(self, key):

        for idx,cls in enumerate(ALL_CLS):
            if key == ord(str(idx)):
                self.mark_cls = cls
        if key == ord("w"):
            if self.box_select_idx>=1:
                self.box_select_idx-=1
        elif key == ord("s"):
            if self.box_select_idx<len(self.labels)-1:
                self.box_select_idx+=1
        elif key == ord("r"):
            self.del_box()
        elif key == ord("`"):
            self.clear_data()

    def img_show(self, img,label=None):
        # while True:
        if not label is None:
            self.labels.extend(label)
        self._show_img = img.copy()
        self.draw()
        cv2.imshow(self.window_name, self._show_img)

    def set_boxs(self, boxs):
        self.labels = boxs

    def draw(self):
        if (self._show_img is None) or (self.labels is None):
            return None

        img_shape = self._show_img.shape
        for box_idx, box in enumerate(self.labels):
            cur_box = box.copy()
            cur_box[:4] = (np.array(cur_box[:4])* (img_shape[1], img_shape[0], img_shape[1], img_shape[0])).astype(np.int)
            if box_idx == self.box_select_idx:
                cv2.rectangle(self._show_img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 0, 255))
                cv2.rectangle(self._show_img, (cur_box[0]+3, cur_box[1]+3), (cur_box[2]-3, cur_box[3]-3), (255, 0, 255))
                cv2.putText(self._show_img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 0, 255))
            else:
                cv2.rectangle(self._show_img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 255, 0))
                cv2.putText(self._show_img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0))

        if self.right_bottom_point != None:
            cv2.rectangle(self._show_img, self.left_top_point, self.right_bottom_point, (0, 0, 255))
            cv2.putText(self._show_img, str(self.mark_cls), (self.left_top_point[0], self.left_top_point[1] + 22),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    def draw_call_back(self, event, x, y, flags, param):
        '''实现了鼠标的画框操作

        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        '''

        (h, w,c) = self._show_img.shape
        if event == cv2.EVENT_LBUTTONDOWN:
            (x, y) = tuple(np.maximum((x, y), (0, 0)))
            (x, y) = tuple(np.minimum((x, y), (w - 1, h - 1)))
            self.left_top_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.left_top_point != None:
                (x, y) = tuple(np.maximum((x, y), (0, 0)))
                (x, y) = tuple(np.minimum((x, y), (w - 1, h - 1)))
                self.right_bottom_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if (self.right_bottom_point != None) and (self.mark_cls != None):
                if (abs(self.right_bottom_point[0] - self.left_top_point[0]) > 10) & (
                        abs(self.right_bottom_point[1] - self.left_top_point[1]) > 10):
                    left_top_point = tuple(np.minimum(self.left_top_point, self.right_bottom_point))
                    right_bottom_point = tuple(np.maximum(self.left_top_point, self.right_bottom_point))
                    call_back_box = [left_top_point[0] / w, left_top_point[1] / h,
                                     right_bottom_point[0] / w, right_bottom_point[1] / h,
                                     self.mark_cls]
                    self.labels.append(call_back_box)
            self.left_top_point = None
            self.right_bottom_point = None

if __name__ == '__main__':
    import os
    import shutil

    duration_data_file_list = data_io.get_file_list("D:\data\smoke_car\RFB用黑烟车数据\\1005mydata/train_data_3/data")
    duration_label_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_3/label"

    window = DurationDataMarkWindow("duration data mark")

    for idx,duration_file_path in enumerate(duration_data_file_list):
        file_name = data_io.get_file_name(duration_file_path)
        label_file_path = os.path.join(duration_label_save_dir,file_name+".txt")
        datas = data_io.video_load(duration_file_path,(1280,720))
        labels = data_io.label_load(label_file_path)
        if labels is None or len(labels) <=0:
            continue
        done = False
        while (not done):
            for data in datas:
                window.img_show(data,labels)
                key = cv2.waitKey(55)
                print(key==113)
                if key == 113:
                    done = True
                    break



