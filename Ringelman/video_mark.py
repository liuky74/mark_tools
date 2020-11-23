import data_io
import cv2
import numpy as np

class SubWindow():
    def __init__(self,window_name = "sub"):
        self.img = None
        self.label = []
        self.window_name = window_name
        # cv2.namedWindow(self.window_name)
        self.done = False
        self.carlib_box=[]
        self.carlib_relative_value=[]
        cv2.setMouseCallback(self.window_name, self.draw_call_back)
        self.box_select_idx=0
        self.do_draw = True
        self.left_top_point = None
        self.right_bottom_point = None

    def compute_carlib(self):
        carlib_box,car_box = self.carlib_box
        lt_point = [(carlib_box[0] - car_box[0])/(car_box[2]-car_box[0]),(carlib_box[1] - car_box[1])/(car_box[3]-car_box[1])]
        carlib_hw_scale = [(carlib_box[3]-carlib_box[1])/(car_box[3]-car_box[1]),(carlib_box[2]-carlib_box[0])/(car_box[2]-car_box[0])]
        self.carlib_relative_value = [[lt_point,carlib_hw_scale]]

    def draw_call_back(self, event, x, y, flags, param):
        '''实现了鼠标的画框操作

        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        '''
        if not self.img is None:
            (h, w, c) = self.img.shape
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
                if (self.right_bottom_point != None):
                    if (abs(self.right_bottom_point[0] - self.left_top_point[0]) > 10) & (
                            abs(self.right_bottom_point[1] - self.left_top_point[1]) > 10):
                        left_top_point = tuple(np.minimum(self.left_top_point, self.right_bottom_point))
                        right_bottom_point = tuple(np.maximum(self.left_top_point, self.right_bottom_point))
                        call_back_box = [left_top_point[0] / w, left_top_point[1] / h,
                                         right_bottom_point[0] / w, right_bottom_point[1] / h,
                                         "carlib"]
                        self.carlib_box = [call_back_box,self.label[self.box_select_idx]]
                        self.compute_carlib()

                self.left_top_point = None
                self.right_bottom_point = None

    def draw(self,img, labels):
        if self.do_draw:
            img_shape = img.shape
            for box_idx, box in enumerate(labels):
                cur_box = box.copy()
                cur_box[:4] = (np.array(cur_box[:4])* (img_shape[1], img_shape[0], img_shape[1], img_shape[0])).astype(np.int)
                if box_idx == self.box_select_idx:
                    cv2.rectangle(img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 0, 255))
                    cv2.rectangle(img, (cur_box[0]+3, cur_box[1]+3), (cur_box[2]-3, cur_box[3]-3), (255, 0, 255))
                    cv2.putText(img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 0, 255))
                else:
                    cv2.rectangle(img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 255, 0))
                    cv2.putText(img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0))

                cur_hw = [cur_box[3] - cur_box[1], cur_box[2] - cur_box[0]]
                for carlib_lt, carlib_hw in self.carlib_relative_value:
                    carlib_lt = [carlib_lt[0] * cur_hw[1], carlib_lt[1] * cur_hw[0]]
                    carlib_hw = [carlib_hw[0] * cur_hw[0], carlib_hw[1] * cur_hw[1]]
                    cv2.rectangle(img,
                                  (int(cur_box[0] + carlib_lt[0]), int(cur_box[1] + carlib_lt[1])),
                                  (int(cur_box[0] + carlib_lt[0] + carlib_hw[1]),
                                   int(cur_box[1] + carlib_lt[1] + carlib_hw[0])),
                                  (0, 255, 0)
                                  )

            if self.right_bottom_point != None:
                cv2.rectangle(img, self.left_top_point, self.right_bottom_point, (0, 0, 255))
                cv2.putText(img, "carlib", (self.left_top_point[0], self.left_top_point[1] + 22),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    def key_process(self,key):
        if key == ord(" "):
            self.done = True
        if key == ord("w"):
            if self.box_select_idx>=1:
                self.box_select_idx-=1
        elif key == ord("s"):
            if self.box_select_idx<len(self.label)-1:
                self.box_select_idx+=1
        elif key == ord("`"):  #清除标记框
            self.carlib_box.clear()
            self.carlib_relative_value.clear()
        elif key == ord("\\"): #切换是否画框
            self.do_draw= not self.do_draw

    def show(self,img,label):
        self.done = False
        self.img = img
        self.label = label
        while(not self.done):
            show_img = self.img.copy()
            self.draw(show_img,self.label)
            cv2.imshow(self.window_name,show_img)
            key = cv2.waitKey(25)
            self.key_process(key)


class ParWindow():
    def __init__(self,window_name = "par"):
        self.img = None
        self.label = []
        self.window_name = window_name
        cv2.namedWindow(window_name)
        self.do_draw=True
        self.carlib_relative_value=[]
        self.sub_w = SubWindow(self.window_name)
        self.done = False
        pass

    def key_process(self,key):
        if key == ord(" "):
            self.sub_w.show(self.img,self.label)
        if key == ord("q"):
            self.done = True
        if key == ord("0"):
            import pickle
            with open("./carlib_relative_value.pkl","wb") as f:
                pickle.dump(self.sub_w.carlib_relative_value,f)


    def draw(self,img, labels):
        img_shape = img.shape[:2]
        for box_idx, box in enumerate(labels):
            cur_box = box.copy()
            cur_box[:4] = (np.array(cur_box[:4]) * (img_shape[1], img_shape[0], img_shape[1], img_shape[0])).astype(
                np.int)
            cur_hw = [cur_box[3] - cur_box[1], cur_box[2] - cur_box[0]]
            for carlib_lt, carlib_hw in self.sub_w.carlib_relative_value:
                carlib_lt = [carlib_lt[0] * cur_hw[1], carlib_lt[1] * cur_hw[0]]
                carlib_hw = [carlib_hw[0] * cur_hw[0], carlib_hw[1] * cur_hw[1]]
                cv2.rectangle(img,
                              (int(cur_box[0] + carlib_lt[0]), int(cur_box[1] + carlib_lt[1])),
                              (int(cur_box[0] + carlib_lt[0] + carlib_hw[1]),
                               int(cur_box[1] + carlib_lt[1] + carlib_hw[0])),
                              (0, 255, 0)
                              )
            cv2.putText(img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0))
            cv2.rectangle(img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 255, 0))

    def show(self,img,label):
        self.img = img
        self.label = label
        show_img = self.img.copy()
        self.draw(show_img,self.label)
        cv2.imshow(self.window_name,show_img)
        key = cv2.waitKey(25)
        self.key_process(key)



if __name__ == '__main__':
    import os
    import shutil
    video_file_list = data_io.get_file_list("D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\smoke_video")
    org_video_label_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\video_labels"
    # video_label_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_3\\video_label"
    # duration_data_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata/train_data_3/data"
    # duration_label_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_3/label"

    par_w = ParWindow()


    video_idx = 0
    for video_file_path in video_file_list:
        par_w.done = False
        print("||video idx: %i|video file name: %s||"%(video_idx,video_file_path))
        video_datas = data_io.video_load(video_file_path, [(1280, 720)])
        file_name = ".".join(os.path.basename(video_file_path).split(".")[:-1])
        # 如果有已经标记完的数据则读取
        label_file_path = os.path.join(org_video_label_dir, file_name + '.txt')
        video_labels = data_io.label_load(label_file_path)
        while(1):
            if par_w.done:
                break
            for frame_idx,video_data in enumerate(video_datas):
                if par_w.done:
                    break
                if frame_idx in video_labels.keys():
                    par_w.show(img=video_data,label=video_labels[frame_idx])


