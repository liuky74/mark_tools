import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

slice_num = 25

def get_threshold_precent(img_HSV, points):
    precents = np.empty(11,np.float)
    points = np.array(points)
    mask = np.zeros(img_HSV.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    pixels = img_HSV[mask > 0][:, 2]
    for idx in range(11):
        precents[idx] = np.percentile(pixels, int((idx) * 10))
        # print("|%d|%f|"%(idx*10,np.percentile(pixels, int((idx) * 10))))
    # print("|diff|%f|"%(np.percentile(pixels, 95)-np.percentile(pixels, 5)))
    # print("|%.2f|%f|" % (0.75, np.percentile(pixels, 15)))
    # print("|%.2f|%f|" % (1, np.percentile(pixels, 20)))
    # print("|%.2f|%f|" % (1.25, np.percentile(pixels, 25)))
    # print("|%.2f|%f|" % (1.5, np.percentile(pixels, 30)))
    # print("|%.2f|%f|" % (1.75, np.percentile(pixels, 35)))
    # print("-------------------------------------------")
    # return np.array([np.percentile(pixels, max(level*20-5,0)),np.percentile(pixels, max(level*20+5,100))])

    return precents

def get_threshold(img_HSV,points,level):
    min_values = np.empty(slice_num,np.int)
    max_values= np.empty(slice_num,np.int)
    points = np.array(points)
    mask = np.zeros(img_HSV.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    pixels = img_HSV[mask > 0][:,2]
    pice_num = pixels.shape[0]//slice_num
    for idx in range(slice_num):
        pice_pixels = pixels[idx*pice_num:(idx+1)*pice_num]
        min_values[idx] = pice_pixels.min()
        max_values[idx] = pice_pixels.max()
    min_value = min_values.mean()
    max_value = max_values.mean()
    diff = max_value-min_value

    threshold = min_value+diff/2



    return threshold


def compute_ringelman(img,img_HSV,points,threshold):
    ringelman_list=[]
    points= np.array(points)
    mask = np.zeros(img.shape[:2],dtype=np.uint8)
    cv2.fillConvexPoly(mask,points,255)
    pixels = img_HSV[mask>0]
    pice_size = int(pixels.shape[0]/slice_num)
    for idx in range(slice_num):
        pice_pixels = pixels[idx*pice_size:idx*pice_size+pice_size,:]
        a = (pice_pixels[:,2]<threshold).sum()
        b = len(pice_pixels)
        if a == 0:
            ringelman_list.append(0)
        else:
            ringelman_list.append(10 * (a / b) / 2)
    ringelman_list.sort()
    res = 0
    num = 0
    for idx in range(int(slice_num*0.3),len(ringelman_list)-int(slice_num*0.3)):
        res+=ringelman_list[idx]
        num+=1
    # print("final:",res/num)
    return res/num


class SubWindow():
    def __init__(self,window_name = "sub"):
        self.img = None
        self.label = []
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.done = False
        self.calib_box=[]
        cv2.setMouseCallback(self.window_name, self.draw_call_back)

        self.move_point = None
        self.mouse_point = (0,0)
        self.level = -1
        self.threshold=-1
        self.clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        self.thresholds = np.empty(shape=(6,11),dtype=np.float)
        self.idx = 0



    def draw_call_back(self, event, x, y, flags, param):
        if not self.img is None:
            (h, w, c) = self.img.shape
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.calib_box)<4:
                    (x, y) = tuple(np.maximum((x, y), (0, 0)))
                    (x, y) = tuple(np.minimum((x, y), (w - 1, h - 1)))
                    self.calib_box.append((x,y))
            elif event == cv2.EVENT_MOUSEMOVE:
                self.mouse_point = (x,y)
                if len(self.calib_box)>0 and len(self.calib_box)<4:
                    self.move_point = (x,y)
                else:
                    self.move_point = None

    def draw(self,show_img):
        for idx in range(1,len(self.calib_box)):
            cv2.line(show_img,self.calib_box[idx-1],self.calib_box[idx],(255,0,0))
        if (not self.move_point is None) and len(self.calib_box)>0:
            cv2.line(show_img,self.calib_box[-1],self.move_point,(255,0,0))
        if len(self.calib_box)>=4:
            cv2.line(show_img, self.calib_box[-1], self.calib_box[0], (255, 0, 0))

            # if self.threshold == -1:
            #     self.threshold = get_threshold(self.img_HSV,self.calib_box)
            # self.level = compute_ringelman(self.img,self.img_HSV,self.calib_box,self.threshold)
            tmp_value = self.thresholds[0].sum()
            precent = get_threshold_precent(self.img_HSV,self.calib_box)

            for idx,threshold in enumerate(self.thresholds):
                value = np.abs(precent - threshold).sum()
                if value<tmp_value:
                    self.level = idx
                    tmp_value = value

        else:
            self.level = -1
        #     self.threshold = -1

        pixel = self.img_HSV[self.mouse_point[1],self.mouse_point[0]]
        cv2.putText(show_img,"|H:%d|S:%d|V:%d|"%(pixel[0],pixel[1],pixel[2]),(10,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 1)
        cv2.putText(show_img,"|Level:%.2f|"%(self.level),(10,80),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 1)
        cv2.putText(show_img,"|Threshold:%d|"%(self.threshold),(10,110),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 1)
        cv2.putText(show_img,"|Idx:%d|"%(self.idx),(10,150),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 1)
    def key_process(self,key):
        if key == ord(" "):
            self.done = True
        if key == ord("r"):
            if len(self.calib_box)>0:
                self.calib_box.pop()
        elif key == ord("d"):
            self.thresholds[self.idx] = get_threshold_precent(self.img_HSV,self.calib_box)
        elif key == ord("w"):
            self.threshold+=2
        elif key == ord("s"):
            self.threshold-=2
        elif key == ord("0"):
            self.idx = 0
        elif key == ord("1"):
            self.idx = 1
        elif key == ord("2"):
            self.idx = 2
        elif key == ord("3"):
            self.idx = 3
        elif key == ord("4"):
            self.idx = 4
        elif key == ord("5"):
            self.idx = 5


    def show(self,img):
        self.done = False
        self.img = img
        self.img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # self.img_HSV[...,2] = self.clahe.apply(self.img_HSV[...,2])
        # self.img_HSV[..., 2] = cv2.equalizeHist(self.img_HSV[..., 2])
        # while(not self.done):
        show_img = self.img.copy()
        self.draw(show_img)
        cv2.imshow(self.window_name,show_img)

        cv2.imshow("tmp",self.img_HSV[...,2])
        key = cv2.waitKey(1)
        self.key_process(key)

if __name__ == '__main__':

    dirs = [x for x in os.walk("./imgs")]
    window = SubWindow()

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for dir in dirs:
        if len(dir[2])>0:
            for filename in dir[2]:
                print(filename)
                img_file_path = os.path.join(dir[0],filename)
                img_data = cv2.imread(img_file_path)
                img_data = cv2.filter2D(img_data, -1, kernel=kernel)
                img_data = cv2.resize(img_data,(640,1080))

                window.show(img_data)


