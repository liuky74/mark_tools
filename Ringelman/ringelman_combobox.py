import tkinter as tk
from tkinter import messagebox,ttk
import os
import numpy as np
import cv2
import pickle

class CBoxRingelman(ttk.Combobox):
    def __init__(self,master):
        super(CBoxRingelman, self).__init__(master)
        self.levels = {"0.75": [],
                       "1.0": [],
                       "1.25": [],
                       "1.5": [],
                       "1.75": [],
                       "2.0": [],
                       "3.0": [],
                       "4.0": [],
                       "5.0": []}
        self['value'] = ("0.75", "1.0", "1.25", "1.5", "1.75", "2.0", "3.0", "4.0", "5.0")
        self.current(0)
        self.configure(state="readonly")
        self.bind("<<ComboboxSelected>>", self.select_handle)
        if os.path.exists("./RingelmanFeatures.pkl"):
            with open("./RingelmanFeatures.pkl","rb") as f:
                self.levels = pickle.load(f)
    def select_handle(self,event):
        print(self.get())

    def save(self):
        if os.path.exists("./RingelmanFeatures.pkl"):
            ret = messagebox.askyesno("提示","特征文件已存在,是否要覆盖?")
            if not ret:
                return
        with open("./RingelmanFeatures.pkl","wb") as f:
            pickle.dump(self.levels,f)

    def get_threshold_precent(self,img_HSV, points):
        if len(points) < 4:
            return None
        precents = np.empty(11, np.float)
        points = np.array(points)
        mask = np.zeros(img_HSV.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        pixels = img_HSV[mask > 0][:, 2]
        if len(pixels)<=0:
            return None
        for idx in range(11):
            precents[idx] = np.percentile(pixels, int((idx) * 10))
        return precents

    def calculate_level(self,img,points):
        level = "-1"
        if len(points)<4:
            return level
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        precent = self.get_threshold_precent(img_HSV, points)
        if precent is None:
            return level
        diff = np.inf

        for (level_,precents_) in self.levels.items():
            if len(precents_)<=0:
                continue
            diff_=0
            for precent_ in precents_:
                diff_ += np.abs(precent - precent_).sum()
            diff_ = diff_/len(precents_)
            if diff_<diff:
                diff = diff_
                level = level_
        return level

    def calib_level(self, img, points):
        if len(self.levels[self.get()])>=4:
            messagebox.showinfo("提示","已达校准上限,请先清空后再校准")
            return
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        precent = self.get_threshold_precent(img_HSV, points)
        self.levels[self.get()].append(precent)
    def pop(self):
        if len(self.levels[self.get()])>0:
            self.levels[self.get()].pop()
        else:
            messagebox.showinfo("提示","数组为空")

    def clear(self):
        self.levels = {"0.75": [],
                       "1.0": [],
                       "1.25": [],
                       "1.5": [],
                       "1.75": [],
                       "2.0": [],
                       "3.0": [],
                       "4.0": [],
                       "5.0": []}


