import data_io
import cv2
import numpy as np
from UI import Window

from configs import ALL_CLS
cls_maps={}
for idx,cls in enumerate(ALL_CLS):
    cls_maps[idx] = cls

X_SUCESS = "x success"

# 生成短视频
def make_ids(frame_idx):
    frame_ids = []
    # for stride in range(12):
    #     # frame_idx-=stride*2
    #     frame_ids.append(frame_idx-stride*2)

    frame_ids.append(frame_idx)
    for stride in range(5):
        frame_idx-= 2
        frame_ids.append(frame_idx)
    for stride in range(4,4+6):
        frame_idx-= stride
        frame_ids.append(frame_idx)


    frame_ids.reverse()
    return frame_ids


class SubWindow():
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

class DurationWindow(Window):
    def __init__(self,window_name,par_window):
        super(DurationWindow, self).__init__(window_name)
        self.par_window = par_window
        self._show_img = None  # np.zeros((720,1280,3),dtype=np.uint8)
        # 高亮框的idx
        self.box_select_idx = 0
        self.video_datas = []
        # label,只会手动清除
        self.label_datas = []
        # 画框过程中保存临时框的坐标
        self.left_top_point = None
        self.right_bottom_point = None
        # 用来保存标记框, 包含下表中的box的目标框将会保留到其他帧中直到手动删除
        self.keep_boxs = []
        # 保留的box,即包含了上面标记框的目标框会保存在这里
        self.keep_label_datas = []
        # 画框的框类别
        self.mark_cls = None
        self.data_save_dir = None
        self.label_save_dir = None

    def menber_init(self,data_save_dir,label_save_dir):
        self.data_save_dir = data_save_dir
        self.label_save_dir = label_save_dir

    def draw(self):
        if (self._show_img is None) or (self.label_datas is None):
            return None
        cv2.putText(self._show_img, str(self.par_window.frame_idx), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 255))
        if self.do_draw:
            img_shape = self._show_img.shape
            for box_idx, box in enumerate(self.label_datas):
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

            if len(self.keep_boxs)>0:
                for keep_boxs in self.keep_boxs:
                    cv2.rectangle(self._show_img, (int(keep_boxs[0]*img_shape[1]),int(keep_boxs[1]*img_shape[0])),
                                  (int(keep_boxs[2]*img_shape[1]),int(keep_boxs[3]*img_shape[0])), (0, 0, 0))
                    cv2.rectangle(self._show_img,
                                  (int(keep_boxs[0] * img_shape[1])+1, int(keep_boxs[1] * img_shape[0])+1),
                                  (int(keep_boxs[2] * img_shape[1])-1, int(keep_boxs[3] * img_shape[0])-1), (255, 255, 255))
    def draw_call_back(self, event, x, y, flags, param):
        '''实现了鼠标的画框操作

        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        '''
        if not self._show_img is None:
            (h, w, c) = self._show_img.shape
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
                        if call_back_box[-1] == -1:#标记点单独保存
                            self.keep_boxs.append(call_back_box)
                        else:
                            self.label_datas.append(call_back_box)
                self.left_top_point = None
                self.right_bottom_point = None

    def del_box(self):
        if (self.label_datas is None) or (len(self.label_datas))==0:
            return None
        for box_idx, box in enumerate(self.label_datas):
            if box_idx == self.box_select_idx:
                self.label_datas.pop(box_idx)
                if self.box_select_idx>0:
                    self.box_select_idx-=1

    def wait_key_process(self, key):

        for idx,cls in enumerate(ALL_CLS):
            if key == ord(str(idx)):
                self.mark_cls = cls
        if key == ord("m"):  # 选择标记,按m后标记出来的框是标记框,所有能包含这个标记框的目标框会被保留,其余删除
            self.mark_cls = -1  # -1表示当前框是标记框
        if key == ord("w"):
            if self.box_select_idx>=1:
                self.box_select_idx-=1
        elif key == ord("s"):
            if self.box_select_idx<len(self.label_datas)-1:
                self.box_select_idx+=1
        elif key == ord("r"):
            self.del_box()
        elif key == ord("`"):  #清除标记框
            self.clear_data()
        elif key == ord(" "):  #恢复暂停
            self.show_done = True
            self.keep_boxs.clear()
        elif key == ord("\\"): #切换是否画框
            self.do_draw= not self.do_draw

        elif key == ord("k"): # 将当前画面的框保持,当切换到别的画面时这些框也会显示
            self.keep_label_datas = self.label_datas.copy()
        elif key == ord("l"): # 取出保持的框数据
            self.keep_label_datas.clear()
        elif key == ord("x"):#跳到最近得下一帧(含有box)
            tmp_frame_idx = self.par_window.frame_idx
            while True:
                tmp_frame_idx+=1
                tmp_frame_idx = tmp_frame_idx % len(self.par_window.video_datas)
                if tmp_frame_idx in self.par_window.label_datas.keys():
                    # data_idxs = [x for x in range(tmp_frame_idx - self.par_window.duration * self.par_window.stride,
                    #                               tmp_frame_idx + self.par_window.stride, self.par_window.stride)]
                    data_idxs = make_ids(tmp_frame_idx)
                    self.loading_data([self.par_window.video_datas[data_idx] for data_idx in data_idxs])
                    self.loading_label(self.par_window.label_datas[tmp_frame_idx].copy())
                    self.par_window.frame_idx = tmp_frame_idx
                    break
        elif key == ord("z"):#跳到最近得上一帧(含有box)
            tmp_frame_idx = self.par_window.frame_idx
            while True:
                tmp_frame_idx-=1
                tmp_frame_idx = tmp_frame_idx % len(self.par_window.video_datas)
                if tmp_frame_idx in self.par_window.label_datas.keys():
                    # data_idxs = [x for x in range(tmp_frame_idx - self.par_window.duration * self.par_window.stride,
                    #                               tmp_frame_idx + self.par_window.stride, self.par_window.stride)]
                    data_idxs = make_ids(tmp_frame_idx)
                    self.loading_data([self.par_window.video_datas[data_idx] for data_idx in data_idxs])
                    self.loading_label(self.par_window.label_datas[tmp_frame_idx].copy())
                    self.par_window.frame_idx = tmp_frame_idx
                    break
        elif key == ord("a"):#上一帧,如果有keep box,则更新母窗口的保存数据
            tmp_frame_idx = self.par_window.frame_idx
            self.update_par_window_labels()
            tmp_frame_idx -= 1
            self.update_datas(tmp_frame_idx)
        elif key == ord("d"):#下一帧
            tmp_frame_idx = self.par_window.frame_idx
            self.update_par_window_labels()
            tmp_frame_idx+=1
            self.update_datas(tmp_frame_idx)
        elif key == ord(","):#删除视频起点到当前帧之前所有帧的label数据
            tmp_frame_idx = self.par_window.frame_idx
            while True:
                tmp_frame_idx-=1
                if tmp_frame_idx <0:
                    break
                if tmp_frame_idx in self.par_window.label_datas.keys():
                    del self.par_window.label_datas[tmp_frame_idx]
        elif key == ord("."):#删除当前帧到视频终点之间所有帧的label数据
            tmp_frame_idx = self.par_window.frame_idx
            while True:
                tmp_frame_idx+=1
                if tmp_frame_idx >len(self.par_window.video_datas):
                    break
                if tmp_frame_idx in self.par_window.label_datas.keys():
                    del self.par_window.label_datas[tmp_frame_idx]
        # 传统训练数据的保存
        elif key == ord("0"):
            if len(self.label_datas) == 0:
                if self.par_window.frame_idx in self.par_window.label_datas.keys():
                    del self.par_window.label_datas[self.par_window.frame_idx]
                else:
                    pass
            else:
                self.par_window.label_datas[self.par_window.frame_idx] = self.label_datas.copy()
            img_shape = self._show_img.shape
            frame_idx = self.par_window.frame_idx
            if frame_idx<24:
                print("frame idx 太小，无法保存")
            else:
                file_name = ".".join(os.path.basename(self.par_window.video_file_path).split(".")[:-1])

                video_file_path = self.data_save_dir+"/%s__%i.mp4"%(file_name,frame_idx)
                label_file_path = self.label_save_dir+"/%s__%i.txt"%(file_name,frame_idx)
                writer = cv2.VideoWriter(video_file_path,cv2.VideoWriter_fourcc(*'DIVX'),25,(img_shape[1],img_shape[0]))
                for img in self.video_datas:
                    writer.write(img)
                writer.release()
                label_f = open(label_file_path,"w",encoding="utf-8")
                for label in self.label_datas:
                    label_str = "%f,%f,%f,%f,%s"%(label[0],label[1],label[2],label[3],label[4])
                    label_f.write(label_str+'\n')
                label_f.close()


    def show(self):
        self.frame_idx = 0
        self.show_done = False
        while True:
            self.frame_idx = self.frame_idx%len(self.video_datas)
            self._show_img = self.video_datas[self.frame_idx].copy()
            self.draw()
            cv2.imshow(self.window_name,self._show_img)
            if self.frame_idx == (len(self.video_datas)-1):
                key = cv2.waitKey(500)
            else:
                key = cv2.waitKey(100)
            self.wait_key_process(key)
            self.frame_idx += 1
            if self.show_done:
                break

    def loading_data(self,data):
        self.video_datas = data

    def loading_label(self,labels):
        self.label_datas = []
        if len(self.keep_label_datas)>0:
            self.label_datas = self.keep_label_datas.copy()
        if self.keep_boxs.__len__()>0:
            for label_data in labels:
                if label_data in self.label_datas:
                    continue
                for keep_boxs in self.keep_boxs:
                    if (label_data[0]<keep_boxs[0] and
                        label_data[1]<keep_boxs[1] and
                        label_data[2]>keep_boxs[2] and
                        label_data[3]>keep_boxs[3]):
                        self.label_datas.append(label_data)
                        break
        else:
            self.label_datas = labels
    #根据帧号更新本windows的数据(包括img和label)
    def update_datas(self,tmp_frame_idx):
        tmp_frame_idx = tmp_frame_idx % len(self.par_window.video_datas)
        # data_idxs = [x for x in range(tmp_frame_idx - self.par_window.duration * self.par_window.stride,
        #                               tmp_frame_idx + self.par_window.stride, self.par_window.stride)]
        data_idxs = make_ids(tmp_frame_idx)
        self.loading_data([self.par_window.video_datas[data_idx] for data_idx in data_idxs])
        if not tmp_frame_idx in self.par_window.label_datas.keys():
            self.loading_label([])
        else:
            self.loading_label(self.par_window.label_datas[tmp_frame_idx].copy())
        self.par_window.frame_idx = tmp_frame_idx

    def update_par_window_labels(self):
        if len(self.keep_boxs) > 0:#在进行前后帧移动时如果没有打keep标记则不对母窗口的label做任何删减
            if len(self.label_datas) == 0:
                if self.par_window.frame_idx in self.par_window.label_datas.keys():
                    del self.par_window.label_datas[self.par_window.frame_idx]
                else:
                    pass
            else:
                self.par_window.label_datas[self.par_window.frame_idx] = self.label_datas.copy()


    def clear_data(self):
        self.keep_boxs=[]
        # self.label_datas.clear()
        # self.left_top_point=None
        # self.right_bottom_point=None
        # self.box_select_idx = 0

class VideoPlayer(Window):
    def __init__(self,window_name):
        super(VideoPlayer,self).__init__(window_name)
        cv2.namedWindow(self.window_name)
        self.duration_window= DurationWindow("video player",self)
        self.org_label_file=None
        self.save_label_file=None
        self.video_file_path = None
        self.org_show = True
        self.do_draw = True

    def menber_init(self,duration_data_save_dir,duration_label_save_dir,delay = 35,duration = 12,stride = 2):
        self.duration_window.menber_init(duration_data_save_dir,duration_label_save_dir)
        self.duration = duration
        self.stride = stride
        self.delay = delay
        self.video_datas = None
        self.label_datas = {}
        self.frame_idx = None
        self._show_img = None


    def draw(self,labels):
        img_shape = self._show_img.shape[:2]
        cv2.putText(self._show_img, str(self.frame_idx), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 255))
        if self.org_show:
            cv2.putText(self._show_img, "org", (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 0, 255))
        if self.do_draw:
            for box_idx, box in enumerate(labels):
                cur_box = box.copy()
                cur_box[:4] = (np.array(cur_box[:4])* (img_shape[1], img_shape[0], img_shape[1], img_shape[0])).astype(np.int)
                cv2.putText(self._show_img, str(cur_box[4]), (cur_box[0], cur_box[1] + 22), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0))
                # if cur_box[4] == "smoke":
                #     cv2.rectangle(self._show_img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (255, 0,0))
                # else:
                cv2.rectangle(self._show_img, (cur_box[0], cur_box[1]), (cur_box[2], cur_box[3]), (0, 255, 0))

    def loading_data(self,resize_shape=None):
        self.video_datas = data_io.video_load(self.video_file_path,resize_shape)
    def loading_label(self,label_file_path):
        self.label_datas = data_io.label_load(label_file_path)

    def saving_label(self,label_file_path):
        data_io.label_save(self.label_datas,label_file_path)

    def get_label(self):
        return self.label_datas
    def wait_key_process(self, key,*inputs):
        if key == ord("a"):
            self.frame_idx -= 5
        elif key == ord("d"):
            self.frame_idx += 5
        elif key == ord(" "):
            # data_idxs = [x for x in range(self.frame_idx-self.duration*self.stride,self.frame_idx+self.stride,self.stride)]
            data_idxs = make_ids(self.frame_idx)
            self.duration_window.loading_data([self.video_datas[data_idx] for data_idx in data_idxs])
            if self.frame_idx in self.label_datas.keys():
                self.duration_window.loading_label(self.label_datas[self.frame_idx].copy())
            else:
                self.duration_window.loading_label([])
            self.duration_window.show()
        elif key == ord("q"):#推出当前视频循环,进入下一个视频
            self.show_done = True
        elif key == ord("`"):
            # for idx in list(self.label_datas.keys()):
            #     for label_idx,label_data in enumerate(self.label_datas[idx]):
            #         if label_data[-1] == "smoke":
            #             self.label_datas[idx].pop(label_idx)
            self.label_datas.clear()
        elif key == ord("="):#重新读取原label文件
            self.loading_label(self.org_label_file)
        elif key == ord("-"):#读取修正后的label文件
            self.loading_label(self.save_label_file)
        elif key == ord("0"):#保存label文件
            data_io.label_save(self.label_datas,self.save_label_file)
        elif key == ord("\\"):#是否显示box数据
            self.do_draw = not self.do_draw

    def show(self):
        self.frame_idx = 0
        self.show_done = False
        while not self.show_done:
            self.frame_idx = self.frame_idx%len(self.video_datas)
            self._show_img = self.video_datas[self.frame_idx].copy()
            labels = []
            if self.frame_idx in self.label_datas.keys():
                labels = self.label_datas[self.frame_idx].copy()

            self.draw(labels)
            cv2.imshow(self.window_name,self._show_img)
            key = cv2.waitKey(35)
            self.wait_key_process(key)
            self.frame_idx += 1


if __name__ == '__main__':
    import os
    import shutil
    video_file_list = data_io.get_file_list("D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\smoke_video")
    org_video_label_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\video_labels_org"
    video_label_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_3\\video_label"
    duration_data_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata/train_data_3/data"
    duration_label_save_dir = "D:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_3/label"

    video_mark = VideoPlayer("video player")
    video_mark.menber_init(duration_data_save_dir=duration_data_save_dir,
                           duration_label_save_dir=duration_label_save_dir)

    video_idx = 0
    for video_file_path in video_file_list:
        if video_idx<20: # smoke car：275
            video_idx+=1
            # shutil.copy(video_file_path,"G:\data\smoke_car/")
            continue
        print("||video idx: %i|video file name: %s||"%(video_idx,video_file_path))
        video_idx+=1
        video_mark.video_file_path = video_file_path
        file_name = ".".join(os.path.basename(video_file_path).split(".")[:-1])
        #如果有已经标记完的数据则读取
        label_file_path = os.path.join(org_video_label_dir, file_name + '.txt')
        if os.path.exists(label_file_path):
            video_mark.org_label_file = label_file_path
            video_mark.loading_label(label_file_path)
        else:
            continue

        label_file_path = os.path.join(video_label_save_dir,file_name+'.txt')
        video_mark.save_label_file = label_file_path
        if os.path.exists(label_file_path):
            video_mark.org_show = False
            video_mark.loading_label(label_file_path)
            # if len(video_mark.label_datas)<=0:
            #     continue
        else:# video labelfile不存在的时候则读取原始数据
            video_mark.org_show = True

        video_mark.loading_data([(300,300),(1280,720)])
        video_mark.show()
