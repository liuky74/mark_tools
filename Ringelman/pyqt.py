import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import numpy as np
import pickle
import os
import torch
import time
from collections import deque

from Ringelman.models.yolo3.yolo_layer import YoLoV3
from Ringelman.models.yolo3.config import cfg,HyperParameter_data as cfg_data


class ImgLabel(QtWidgets.QLabel):
    def __init__(self, parent):
        super(ImgLabel, self).__init__(parent)
        self.calib_box = []
        self.mouse_point = QtCore.QPoint()
        self.setMouseTracking(True)
        self.setStyleSheet("QLabel{background:white;}")

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        if len(self.calib_box) <= 0:
            return
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setPen(QtGui.QColor(255, 0, 0))
        for idx in range(1, len(self.calib_box)):
            painter.drawLine(self.calib_box[idx], self.calib_box[idx - 1])
        if len(self.calib_box) < 4:
            painter.drawLine(self.mouse_point, self.calib_box[-1])
        else:
            painter.drawLine(self.calib_box[-1], self.calib_box[0])

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.calib_box.append(QtCore.QPoint(ev.x(), ev.y())) if len(self.calib_box) < 4 else None
        self.update()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.mouse_point.setX(ev.x())
        self.mouse_point.setY(ev.y())
        self.update()


class Video():
    def __init__(self, size):
        self.size = size
        self.video_file_path = None
        self.cap = None

    def load_file(self, file_path):
        self.video_file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        return self.cap.isOpened()

    def player(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.cap = cv2.VideoCapture(self.video_file_path)
            ret, frame = self.cap.read()
        # frame = cv2.resize(frame, self.size)

        return frame


class Camera():
    def __init__(self, size):
        self.size = size
        self.rtsp = None
        self.cap = None

    def load_camera(self, rtsp):
        self.rtsp = rtsp
        self.cap = cv2.VideoCapture(rtsp)
        return self.cap.isOpened()

    def player(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("读取画面失败")
        frame = cv2.resize(frame, self.size)
        return frame


class Ringelman(QtWidgets.QComboBox):
    def __init__(self, parent):
        super(Ringelman, self).__init__(parent)
        self.levels = {"0.75": None,
                       "1": None,
                       "1.25": None,
                       "1.5": None,
                       "1.75": None,
                       "2": None,
                       "3": None,
                       "4": None,
                       "5": None}
        if os.path.exists("./RingelmanFeatures.pkl"):
            with open("./RingelmanFeatures.pkl","rb") as f:
                self.levels = pickle.load(f)
        self.addItems(["0.75", "1", "1.25", "1.5", "1.75", "2", "3", "4", "5"])

    def level_clear(self):
        self.levels = {"0.75": None,
                       "1": None,
                       "1.25": None,
                       "1.5": None,
                       "1.75": None,
                       "2": None,
                       "3": None,
                       "4": None,
                       "5": None}

    @staticmethod
    def get_threshold_precent(img_HSV, points):
        if len(points) < 4:
            return
        precents = np.empty(11, np.float)
        points = np.array(points)
        mask = np.zeros(img_HSV.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        pixels = img_HSV[mask > 0][:, 2]
        for idx in range(11):
            precents[idx] = np.percentile(pixels, int((idx) * 10))
        return precents

    def calculate_level(self,img,qtpoints):
        if len(qtpoints)<4:
            return
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        points = []
        for point in qtpoints:
            points.append([point.x(), point.y()])
        precent = Ringelman.get_threshold_precent(img_HSV, points)
        diff = np.inf
        level = "-1"
        for (level_,precent_) in self.levels.items():
            if precent_ is None:
                continue
            if np.abs(precent - precent_).sum()<diff:
                diff = np.abs(precent - precent_).sum()
                level = level_
        cv2.putText(img, "|Level:%s|" % (level), (10, 150),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    def calib_level(self, img, qtpoints):
        if not self.levels[self.currentText()] is None:
            ret = QtWidgets.QMessageBox.question(self, "确认", "是否覆盖当前校定等级?",
                                                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                 QtWidgets.QMessageBox.No)
            if ret == QtWidgets.QMessageBox.No:
                return
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        points = []
        for point in qtpoints:
            points.append([point.x(), point.y()])
        precent = Ringelman.get_threshold_precent(img_HSV, points)
        self.levels[self.currentText()] = precent


from Ringelman.models.yolo3.loss.pred_transform import PredTransform
from Ringelman.models.yolo3.utils import ResImgProcess

class CarDetect():
    def __init__(self,output_size):
        self.model = YoLoV3(cfg).eval()
        self.model.load_state_dict(torch.load("./Base320_D1_FOCAL_yolov3_C3_E180.snap")["model"])
        self.model.cuda()
        self.pred_trans = PredTransform(cfg,output_size)
        self.pred_process = ResImgProcess(cfg,cfg_data)
        self.pred_res = None

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




from collections import deque
from threading import Thread,Lock
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(1440, 768)
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        self.frameGeometry().moveCenter(cp)
        self.setWindowTitle("Ringelman")
        self.frame = None
        self.detect_model = CarDetect((1280,720))


        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.play)
        self._timer.start(1)

        self.video = Video((1280, 720))
        self.camera = Camera((1280,720))

        self.pic_box = ImgLabel(self)
        self.pic_box.setFixedSize(1280, 720)
        self.pic_box.setText("显示框")
        self.pic_box.move(150, 24)

        btn_open_file = QtWidgets.QPushButton(self)
        btn_del_calib_point = QtWidgets.QPushButton(self)
        btn_calib_level = QtWidgets.QPushButton(self)
        btn_level_clear = QtWidgets.QPushButton(self)
        self.ringelman_combobox = Ringelman(self)
        txt_1 = QtWidgets.QLabel("标定等级:", self)
        btn_save_features = QtWidgets.QPushButton(self)


        btn_open_file.resize(100, 30)
        btn_del_calib_point.resize(100, 30)
        btn_calib_level.resize(100, 30)
        btn_save_features.resize(100,30)
        btn_level_clear.resize(100,30)


        self.ringelman_combobox.resize(100, 30)

        btn_open_file.setText("打开视频")
        btn_del_calib_point.setText("删除点(&r)")
        btn_calib_level.setText("进行校定(&d)")
        btn_save_features.setText("保存校订文件(%s)")
        btn_level_clear.setText("清空校订")

        btn_open_file.move(5, 5)
        btn_del_calib_point.move(5, 40)
        txt_1.move(5, 75)
        self.ringelman_combobox.move(5, 95)
        btn_calib_level.move(5, 130)
        btn_save_features.move(5,165)
        btn_level_clear.move(5,200)

        btn_open_file.clicked.connect(self.open)
        btn_del_calib_point.clicked.connect(self.remove_calib_point)
        # self.ringelman_combobox.currentIndexChanged[str].connect(self.ringelman.set_cur_level)
        btn_calib_level.clicked.connect(self.calib_level)
        btn_save_features.clicked.connect(self.save_features)
        btn_level_clear.clicked.connect(lambda :self.ringelman_combobox.level_clear())
        self.update()


        #维护变量
        self.box_idx=0


    def car_detect(self):
        self.detect_model(self.frame)



    def save_features(self):
        with open("./RingelmanFeatures.pkl","wb") as f:
            pickle.dump(self.ringelman_combobox.levels,f)


    def calib_level(self):
        if len(self.pic_box.calib_box) < 4 or self.frame is None:
            return
        self.ringelman_combobox.calib_level(self.frame, self.pic_box.calib_box)

    def remove_calib_point(self):
        if len(self.pic_box.calib_box) > 0:
            self.pic_box.calib_box.pop()
            self.pic_box.update()

    def open(self, even):
        imgName, imgType = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "D:\data\\tmp")
        if len(imgName)<=0:
            return
        ret = self.video.load_file(imgName)

        # ret = self.camera.load_camera(0)

        if not ret:
            print("打开文件失败")

    def opencv_draw(self):
        self.ringelman_combobox.calculate_level(self.frame, self.pic_box.calib_box)

        #车辆画框
        if len(self.detect_model.pred_res[0])<=0 or len(self.detect_model.pred_res[1])<=0:
            pass
        else:
            for box_idx,(box,label) in enumerate(zip(self.detect_model.pred_res[0].detach().cpu().numpy().astype(np.int),
                                 self.detect_model.pred_res[1].detach().cpu().numpy().astype(np.int))):
                if box_idx == self.box_idx:
                    color = (0,0,255)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), color)
                cv2.putText(self.frame, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)

        key = cv2.waitKey(1)
        if key == ord("w"):
            self.box_idx-=1
            self.box_idx = max(self.box_idx,0)

        elif key == ord("s"):
            self.box_idx += 1
            self.box_idx = min(self.box_idx, len(self.detect_model.pred_res[0]))
        # self.detect_model.pred_process.process(self.detect_model.pred_res,self.frame)


    def play(self):
        frame = self.video.player()
        # frame = self.camera.player()
        if frame is None:
            return
        self.frame = frame
        # 深度模型车辆检测
        self.car_detect()
        #opencv 画图
        self.opencv_draw()


        #pyqt画图
        frame_qt = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1] * self.frame.shape[2],
                                QtGui.QImage.Format_BGR888)
        self.pic_box.setPixmap(QtGui.QPixmap.fromImage(frame_qt))
        self.pic_box.update()


if __name__ == '__main__':
    # show_qt()

    app = QApplication(sys.argv)

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
