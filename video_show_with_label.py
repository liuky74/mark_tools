# NNIE测试时使用,读取测试视频并加载每一帧的label显示

import data_io

def load_NNIE_label_file(path):
    # path = "D:\label_files\\video_img_23.txt"
    labels = []
    with open(path,"r") as f:
        readlines = f.readlines()
        for readline in readlines:
            readline = readline.strip()
            readline_splits = readline.split(" ",3)
            boxs = readline_splits[-1][1:-1].split(",")
            boxs = [float(x)/300 for x in boxs][:4]
            cls = int(readline_splits[1][-2])
            labels.append(boxs+[cls])
    return labels







if __name__ == '__main__':
    import cv2
    import os
    label_files = data_io.get_file_list("D:/label_files")
    video_datas = data_io.video_load("test_video_2.mp4", (1280, 720))

    ratio = (1280,720)

    for idx,video_data in enumerate(video_datas):
        label_file_path = "D:/label_files/video_img_%i.txt"%idx
        if os.path.exists(label_file_path):
            labels = load_NNIE_label_file(label_file_path)
        else:
            labels = []

        for label in labels:
            pt1 = (int(label[0]*ratio[0]),int(label[1]*ratio[1]))
            pt2 = (int(label[2]*ratio[0]),int(label[3]*ratio[1]))
            cv2.rectangle(video_data,pt1,pt2,(255,0,0))
        cv2.imshow("video show",video_data)
        cv2.waitKey(25)

