cfg = {
    "cuda_device":0,
    "batch_size":1,
    "num_class": 4,
    "input_size": 320,
    "in_channel":3,
    # anchor是以512为基准的，如果改变input_size需要自行调整权重anchor_scale
    "anchors": [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ],
    "anchor_scale":0.7,
    # "feature_maps": [52, 26, 13],# 必须为input_size的1/8，1/16，1/32，且必须能除尽，否则无法对齐
    "feature_maps": [40, 20, 10],
    "steps": [8, 16, 32],
    # "batch_size": 8,
    "wh_iou":0.2,
    "pos_weight":None,
    "cls_weight":None,
    "loss_fun":"BCE",

}

HyperParameter = {

    "weight_file_path": "weights/snaps/Base320_D1_FOCAL_yolov3_C3_E180.snap",#会根据模型的名字自动选择RFB或者yolo3
    #RFB320_D12_CE_yolov3_C5_E300 Base416_D12_BCE_yolov3_C1_E260
    "input_size": 320,
    "anchor_scale": 0.625,
    "in_channel":3,
    "duration": 1,
    "grade_data":False,
    "Threshold":0.5,
    "duration_threshold":5,
    "num_class": 3,
    "cuda_device": 0,
    "half": False,
    "main_clses":[0,],  # 主要检测类,会触发检测判定
    "other_clses":[3,4,],  # 附属检测类,不会触发检测判定 不在主副类内的目标不会画框
}

feature_maps = {"feature_maps": [HyperParameter["input_size"] // 8, HyperParameter["input_size"] // 16,
                                 HyperParameter["input_size"] // 32]}

cfg.update(feature_maps)
cfg.update(HyperParameter)

HyperParameter_data={
    "data_dir": "D:\data\smoke_car\RFB用黑烟车数据\测试数据\MP4-20190531\T",
    # "data_dir":"/home/liuky/HDD_1/data/smoke/train_data/1005smokedata/seq12_data/video_grade8",
    # "data_dir": '/home/liuky/HDD_1/data/smoke/val_data_2',
    "save_video": False,
    "img_size":[1280,720],
    "T_save_dir": "D:/data/tmp/T",
    "F_save_dir": "D:/data/tmp/F",
    "draw_box":True,
    "show":False,
}