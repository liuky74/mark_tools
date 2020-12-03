import tkinter as tk
from tkinter import filedialog,messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from Ringelman.ringelman_combobox import CBoxRingelman
from Ringelman.car_detect import CarDetect
from Ringelman.data_box import DataBox

class ShowCanvas(tk.Canvas):
    def __init__(self,master,top,width,height,bg="snow"):
        super(ShowCanvas, self).__init__(master=master,width=width,height=height,bg=bg)
        self.top = top
        self.show_img = None
        self.bind("<Button-1>",self.mouse_press)
        self.bind("<Motion>", self.mouse_move)
        self.calib_box=[]
        self.move_point = []
        self.img_handlers=[] #保存canvas生成图片的句柄


    def points_clear(self):
        self.calib_box = []
        self.move_point = []

    def mouse_press(self,event):
        if self.top.pause:
            if len(self.calib_box)<4:
                self.calib_box.append([event.x,event.y])
    def mouse_move(self,event):
        if self.top.pause:
            self.move_point = [event.x,event.y]

    def draw_text(self,img,text,point,size,color=(0,0,255)):
        cv2.putText(img,text,point,cv2.FONT_HERSHEY_COMPLEX, size,color)

    def draw_box(self,img):

        color = (0, 0, 255)
        for idx,point in enumerate(self.calib_box):#多边形画线
            if idx == 0: continue
            cv2.line(img,(point[0],point[1]),(self.calib_box[idx-1][0],self.calib_box[idx-1][1]),color=color)

        if len(self.calib_box)>=4:
            cv2.line(img, (self.calib_box[-1][0],self.calib_box[-1][1]),(self.calib_box[0][0],self.calib_box[0][1]), color=color)
        else:
            if len(self.calib_box)>0:
                cv2.line(img, (self.move_point[0],self.move_point[1]),(self.calib_box[-1][0],self.calib_box[-1][1]), color=color)

    def img_handlers_clear(self):
        for x in range(len(self.img_handlers)//2):
            self.delete(self.img_handlers[x])



class MainWindow(tk.Frame):
    def __init__(self,master):
        super(MainWindow, self).__init__(master)
        self.place(x = 0,y = 0)
        self.cap = None
        self.pause = False
        self.frame = np.zeros((720,1280,3),np.uint8)*255
        self.show_runing=False
        self.detect_module = CarDetect(top = self,output_size = (1280,720))

        self.init(master)

    def init(self,master):
        self.bind_all("<KeyPress>",self.key_press_process)

        self.rbtn_var = tk.IntVar()
        self.rbtn_var.set(0)
        self.rbtn_polygon = tk.Radiobutton(master, text = 'static', variable = self.rbtn_var, value = 0,command = self.rbtn_handler)
        self.rbtn_polygon.place(x = 10,y = 150)
        self.rbtn_rectangle = tk.Radiobutton(master,text = 'dynamic',variable = self.rbtn_var,value = 1,command = self.rbtn_handler)
        self.rbtn_rectangle.place(x=10,y=170)

        self.btn_open_file = tk.Button(master,text="打开文件",command = self.open_file,padx=20,pady = 0)
        self.btn_open_file.place(x=10,y=10)
        self.btn_open_file = tk.Button(master, text="打开摄像头", command=self.open_camera, padx=12, pady=0)
        self.btn_open_file.place(x=110, y=10)
        self.btn_open_file = tk.Button(master, text="校准", command=self.carlibration, padx=30, pady=0)
        self.btn_open_file.place(x=10, y=75)
        self.btn_open_file = tk.Button(master, text="清除", command=self.pop_rgm_feature, padx=0, pady=0)
        self.btn_open_file.place(x=180, y=45)
        self.btn_open_file = tk.Button(master, text="初始化", command=self.clear_rgm_feature, padx=24, pady=0)
        self.btn_open_file.place(x=110, y=75)
        self.btn_open_file = tk.Button(master, text="保存校准文件", command=self.save_rgm_feature, padx=56, pady=0)
        self.btn_open_file.place(x=10, y=105)

        self.cbox_ringelman = CBoxRingelman(master)
        self.cbox_ringelman.place(x = 10,y = 45)

        self.show_canvas = ShowCanvas(master,top = self, width = 1280, height = 720)
        self.show_canvas.place(x =1500 - 1280, y=768 - 720)
        self.show_canvas_show()
        self.show_runing = True

    def rbtn_handler(self):#切换模式时自动清理canvas中的点数据
        self.show_canvas.points_clear()

    def carlibration(self):
        self.cbox_ringelman.calib_level(self.frame,self.show_canvas.calib_box)
    def clear_rgm_feature(self):
        self.cbox_ringelman.clear()
    def pop_rgm_feature(self):
        self.cbox_ringelman.pop()
    def save_rgm_feature(self):
        self.cbox_ringelman.save()


    def key_press_process(self,event):
        if event.keysym == 'r':
            if len(self.show_canvas.calib_box) > 0:
                self.show_canvas.calib_box.pop()
        elif event.keysym == 'space':
            self.pause = bool(1-self.pause)
            self.show_canvas.pause = self.pause
        elif event.keysym == 'w':
            self.detect_module.select_idx -=1
            self.detect_module.select_idx =max(self.detect_module.select_idx,0)
        elif event.keysym == 's':
            self.detect_module.select_idx +=1
            self.detect_module.select_idx =min(self.detect_module.select_idx,len(self.detect_module.pred_res[0])-1)
        else:
            print(event.keysym)
            return

    def show_canvas_show(self):
        if self.cap is None:
            pass
        else:
            if self.pause:
               pass
            else:
                ret,frame = self.cap.read()
                if not ret:
                    self.cap = cv2.VideoCapture(self.file_path)
                    ret,frame = self.cap.read()
                    if not ret:
                        messagebox.showinfo("错误",message="读取文件失败")
                        self.show_runing = False
                        return
                self.frame = cv2.resize(frame,(1280,720))
        # 调用模型找出车辆
        self.detect_module(self.frame)
        # 使用opencv画图
        show_frame= self.frame.copy()
        self.show_canvas.draw_box(show_frame)
        self.detect_module.draw_box(show_frame)

        level = self.cbox_ringelman.calculate_level(self.frame, self.show_canvas.calib_box)
        self.show_canvas.draw_text(show_frame,"|Level:%s|"%level,(20,40),1)
        self.show_canvas.show_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(show_frame,cv2.COLOR_BGR2RGB)))
        self.show_canvas.img_handlers.append(self.show_canvas.create_image(0, 0, anchor = tk.NW, image = self.show_canvas.show_img))

        self.update_idletasks()
        self.update()
        self.show_canvas.img_handlers_clear()
        self.show_canvas.after(int(1), self.show_canvas_show)

        print(self.rbtn_var.get())


    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showinfo("提示",message="打开摄像头失败")
            self.cap = None
        else:
            if not self.show_runing:
                self.show_canvas_show()
                self.show_runing = True

    def open_file(self):
        file_path= filedialog.askopenfilename(title=u'选择文件')
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showinfo("提示",message="打开视频文件失败")
            self.cap = None
        else:
            if not self.show_runing:
                self.show_canvas_show()
                self.show_runing = True






if __name__ == '__main__':
    top = tk.Tk()
    top.grid(1500,768,1500,768)
    a = MainWindow(top)
    top.mainloop()


