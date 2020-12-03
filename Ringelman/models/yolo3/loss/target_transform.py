import torch
import numpy as np


def toxywh(lr_point):
    wh = lr_point[:,[2,3]] - lr_point[:,[0,1]]
    cent_xy = (lr_point[:,[0,1]]+lr_point[:,[2,3]])/2
    xywh = torch.cat([cent_xy,wh],-1)
    return xywh

def fun_wh_iou(wh1, wh2):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2)
    inter = inter.prod(2)
    wh_iou = inter / (wh1.prod(2) + wh2.prod(2) - inter)
    #a=wh_iou.argmax(axis=1)
    # wh_iou[range(wh_iou.shape[0]),wh_iou.argmax(axis=1)]=2.0
    return wh_iou
def wh_iou_statistic(wh_ious):
    for idx in range(wh_ious[0].shape[0]):
        wh_iou = torch.cat([wh_ious[i][idx,...] for i in range(len(wh_ious))])
        max_idx = wh_iou.argmax()
        map_idx = max_idx // len(wh_ious)
        iou_idx = max_idx % wh_ious[map_idx].shape[-1]
        wh_ious[map_idx][idx,iou_idx] = 2.0
    return wh_ious


class TargetTransform():
    def __init__(self,cfg):
        self.map_sizes = cfg["feature_maps"]
        self.cuda_device = cfg["cuda_device"]
        self.steps = cfg["steps"]
        self.total_anchors = (np.array(cfg["anchors"])*cfg["anchor_scale"]).astype(np.float32)
        self.total_anchors_vec = self.build_anchor_vecs()
        self.wh_iou_th = cfg["wh_iou"]
    def __call__(self,targets):
        # target_idx
        # target_bbox
        # target_cls
        res_idx=[]
        res_bbox=[]
        res_cls=[]
        res_anchor_vec=[]

        wh_ious = []
        anchor_idxs=[]
        mid_targets=[]
        for map_size,anchor_vec in zip(self.map_sizes,self.total_anchors_vec):
            scale = torch.Tensor([map_size,map_size,map_size,map_size,1,1]).float()
            # scale = scale.cuda()
            # anchor_vec = anchor_vec.cuda()
            target = targets*scale
            # LRpoint 转 xywh
            target[:,:4] = toxywh(target[:,:4])
            # 将anchor放大缩小到当前map的scale
            anchor_idx = torch.arange(0, anchor_vec.shape[0]).unsqueeze(0).expand([targets.shape[0],-1]).reshape([-1,1])
            # 计算wh iou，将小于阈值的bbox去掉xx.
            wh_iou = fun_wh_iou(target[:, 2:4], anchor_vec)
            #保存中间数据
            mid_targets.append(target)
            anchor_idxs.append(anchor_idx)
            wh_ious.append(wh_iou)

        if self.wh_iou_th > 0:  # 如果不进行wh iou的筛选，可能会导致感受野大的pred_box强行匹配感受野小的target_box
            wh_ious = wh_iou_statistic(wh_ious)

        for target,anchor_idx,wh_iou,anchor_vec in zip(mid_targets,anchor_idxs,wh_ious,self.total_anchors_vec):
            # 将target按照anchor的数量进行扩展
            target = target.unsqueeze(1).expand([target.shape[0], anchor_vec.shape[0], -1]).reshape([-1, target.shape[-1]])
            wh_iou = wh_iou.view(-1)
            if self.wh_iou_th > 0:  # 如果不进行wh iou的筛选，可能会导致高感受野的pred_box强行匹配低感受野的target_box
                anchor_idx = anchor_idx[wh_iou >= self.wh_iou_th]
                target = target[wh_iou >= self.wh_iou_th, ...]
            if target.shape[0] == 0:  # 当前feature size大小下所有target box都不匹配（不满足wh_iou）
                # print("target.shape[0]==0")
                res_anchor_vec.append(None)
                res_bbox.append(None)
                res_idx.append(None)
                res_cls.append(None)
                continue

            # 得到每个bbox对应的anchor单位向量
            anchor_vecs = anchor_vec[anchor_idx.view(-1)]
            #整合数据
            img_idx = target[:,-1].unsqueeze(-1)
            grid_xy = torch.floor(target[:,:2])
            grid_yx = grid_xy[:,[1,0]]
            bias_xy = target[:,:2]-grid_xy
            wh = target[:,2:4]
            bbox = torch.cat([bias_xy,wh],-1)
            idx = torch.cat([img_idx.long(),grid_yx.long(),anchor_idx.long()],-1)
            cls = target[:, -2]
            #判断是否使用cuda
            if self.cuda_device!=-1:
                anchor_vecs = anchor_vecs.cuda(self.cuda_device)
                bbox=bbox.cuda(self.cuda_device)
                idx=idx.cuda(self.cuda_device)
                cls = cls.cuda(self.cuda_device)
            #保存数据
            res_anchor_vec.append(anchor_vecs)
            res_bbox.append(bbox)
            res_idx.append(idx)
            res_cls.append(cls)
        return {'idx':res_idx,'cls':res_cls,'bbox':res_bbox,'anchor_vec':res_anchor_vec}

    def build_anchor_vecs(self):
        total_anchors_vec=[]
        for anchors,step in zip(self.total_anchors,self.steps):
            total_anchors_vec.append(torch.from_numpy(anchors/step).float())
        return total_anchors_vec









if __name__ == '__main__':
    import pickle
    from models.yolo3.config import cfg
    with open("yolo_target.pkl","rb") as f:
        target = pickle.load(f)
    target = target[:,[2,3,4,5,1,0]]
    target = torch.from_numpy(target)
    TargetTransform(cfg)(target)