from .focal_loss import FocalLoss
# from models.yolo3.config import cfg
from .target_transform import TargetTransform

def bbox_giou(box1, box2, x1y1x2y2=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = box1.t()
    box2 = box2.t()

    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c_area = cw * ch + 1e-16  # convex area
    return iou - (c_area - union) / c_area  # GIoU


class MultiBoxLoss():
    def __init__(self, cfg):
        self.map_sizes = cfg["feature_maps"]
        self.steps = cfg["steps"]
        self.total_anchors = cfg["anchors"]
        self.cuda_device = cfg["cuda_device"]
        self.target_transform = TargetTransform(cfg)
        if cfg["loss_fun"] == "BCE":
            self.BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=cfg['cls_weight'], reduction="mean")
            self.BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=cfg['pos_weight'], reduction="mean")
        elif cfg["loss_fun"] == "FOCAL" :
            self.BCEobj = FocalLoss(pos_weight=cfg['cls_weight'], reduction="mean")
            self.BCEcls = FocalLoss()
        else:
            print("wrong")
            exit(-1)

    def __call__(self, pred_logit, target):
        # pred_logit: [map_num,[batch_size,grid_y,grid_x,anchor_num,[bias_x,bias_y,w,h,obj,clses...]]]
        cls_loss = torch.Tensor([0]).float()
        box_loss =  torch.Tensor([0]).float()
        obj_loss =  torch.Tensor([0]).float()
        if self.cuda_device!=-1:
            cls_loss = cls_loss.cuda(self.cuda_device)
            box_loss = box_loss.cuda(self.cuda_device)
            obj_loss = obj_loss.cuda(self.cuda_device)
        # 当所有target中都不包含目标时,只进行obj的loss计算,其余两个loss为0
        if (target.shape[0]==0):
            target_idxs=[None,None,None]
        else:
            target = self.target_transform(target)
            target_idxs = target['idx']
            # target_idxs = [target_idxs[i] for i in range(len(target_idxs)-1,-1,-1)]
            target_clses = target['cls']
            target_bboxes = target['bbox']
            target_anchor_vecs = target['anchor_vec']
        for idx in range(len(pred_logit)):# feature_map idx
            pred_ = pred_logit[idx]
            target_idx = target_idxs[idx]
            # 当前feature size没有合适的target box，因此只计算obj_cls
            if target_idx is None:
                target_obj = torch.zeros(pred_.shape[:-1], dtype=torch.float32)#没有obj，全部均为0
                if self.cuda_device != -1:
                    target_obj = target_obj.cuda(self.cuda_device)
                pred_obj = pred_[..., 4]
                single_obj_loss = self.BCEobj(pred_obj, target_obj)
                obj_loss += single_obj_loss
            else:
                img_idx, grid_y, grid_x, anchor_idx = target_idx.split(1, -1)
                pred_pos = torch.squeeze(pred_[img_idx, grid_y, grid_x, anchor_idx],dim=1)

                # box loss计算
                pred_xy = torch.sigmoid(pred_pos[..., :2])
                pred_wh = torch.exp(pred_pos[..., 2:4]).clamp(0, 1e3) * target_anchor_vecs[idx]
                pred_bbox = torch.cat([pred_xy, pred_wh], axis=-1)
                # g_iou = fun_giou(pred_bbox, target_bboxes[idx])
                g_iou = bbox_giou(pred_bbox, target_bboxes[idx])
                single_box_loss = torch.mean(1 - g_iou)
                box_loss += single_box_loss

                # obj&cls loss计算
                # one hot
                pred_cls = pred_pos[:, 5:]
                target_cls = torch.zeros(pred_cls.shape, dtype=torch.float32)
                target_obj = torch.zeros(pred_.shape[:-1], dtype=torch.float32)
                if self.cuda_device!=-1:
                    target_cls = target_cls.cuda(self.cuda_device)
                    target_obj = target_obj.cuda(self.cuda_device)
                # obj loss
                target_obj[img_idx, grid_y, grid_x, anchor_idx] = 1
                pred_obj = pred_[..., 4]
                single_obj_loss = self.BCEobj(pred_obj, target_obj)
                obj_loss +=single_obj_loss
                # cls loss
                tmp_idx1=torch.arange(0, target_cls.shape[0])
                tmp_idx2=target_clses[idx].to(torch.long)
                target_cls[tmp_idx1,tmp_idx2]=1
                single_cls_loss =self.BCEcls(pred_cls,target_cls)
                cls_loss += single_cls_loss

        return box_loss,obj_loss,cls_loss


if __name__ == '__main__':
    import pickle
    import torch
    import numpy as np

    with open("yolo_target.pkl", "rb") as f:
        target = pickle.load(f)
    with open("pred.pkl", "rb") as f:
        preds = pickle.load(f)
        preds = [np.transpose(pred, [0, 2, 3, 1, 4]) for pred in preds]
    preds = [torch.Tensor(preds[i]).float() for i in range(len(preds) - 1, -1, -1)]
    target = target[:, [2, 3, 4, 5, 1, 0]]
    target = torch.from_numpy(target)
    MultiBoxLoss(cfg)(preds, target)
