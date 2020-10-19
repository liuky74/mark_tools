# 检查label文件与data文件是否一一对应

import os
label_dir= "G:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_2\label"
data_dir="G:\data\smoke_car\RFB用黑烟车数据\\1005mydata\\train_data_2\data"

label_file_names = [x for x in os.walk(label_dir)][0][2]
data_file_names = [x for x in os.walk(data_dir)][0][2]

label_file_names = [".".join(x.split(".")[:-1]) for x in label_file_names ]
data_file_names = [".".join(x.split(".")[:-1]) for x in data_file_names ]

for label_name in label_file_names:
    if not label_name in data_file_names:
        print(label_name)

for data_name in data_file_names:
    if not data_name in label_file_names:
        print(data_name)


print('')


