import os
import numpy as np
import cv2
def get_file_list(path,extentions=None):
    '''根据后缀名获得目录下的对应文件集

    :param path:
    :param extentions:
    :return:
    '''
    file_dirs = [x for x in os.walk(path)]
    files_list = []
    for file_dir in file_dirs:
        if len(file_dir[2]) > 0:
            for filename in file_dir[2]:
                extention = filename.split('.')[-1]
                if extentions!=None:
                    if not extention in extentions:
                        continue
                file_path = os.path.join(file_dir[0], filename)
                files_list.append(file_path)
    return files_list

def video_load(path,resize = None):
    video_datas=[]
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        if not resize is None:
            for s in resize:
                frame = cv2.resize(frame,s)
        video_datas.append(frame)
    return video_datas

def get_file_name(file_path):
    file_name = ".".join(os.path.basename(file_path).split(".")[:-1])
    return file_name

def label_load(path,shape = None):
    total_labels = {}
    if not os.path.exists(path):
        print("修改后的label文件不存在")
        return total_labels

    with open(path,"r",encoding="utf-8") as f:
        readlines = f.readlines()
        labels = []
        frame_idx = None
        for line_idx,readline in enumerate(readlines):
            if "frame idx" in readline:
                if not frame_idx is None:
                    total_labels[frame_idx] = labels.copy()
                    labels.clear()
                frame_idx = int(readline.split(":")[-1])
                continue
            label = readline.split(",")
            label = [x.strip() for x in label]
            cls_str = label[-1]
            box = np.array([float(x) for x in label[:-1]])
            if not shape is None:
                box = box * (shape[0],shape[1],shape[0],shape[1])
                box = box.astype(np.int)
            label = box.tolist()
            label.append(cls_str)
            labels.append(label)
    if frame_idx is None and len(labels)>0:
        return labels
    return total_labels

def label_save(total_labels,path):
    keys = total_labels.keys()
    # if len(keys) == 0:
    #     return
    with open(path,"w",encoding="utf-8") as f:
        for frame_idx in keys:
            f.write("frame idx:%i\n"%frame_idx)
            labels = total_labels[frame_idx]
            for label in labels:
                label_str = "%s,%s,%s,%s,%s\n"%(str(label[0]),str(label[1]),str(label[2]),str(label[3]),label[4])
                f.write(label_str)


if __name__ == '__main__':
    label_load("tmp_label.txt")



