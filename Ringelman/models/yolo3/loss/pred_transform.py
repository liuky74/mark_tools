import torch
import numpy as np
from Ringelman.models.utils.nms import nms_pytorch

def xywh2LRpoint(boxes):
    l_xy=boxes[:,:2]-boxes[:,2:]/2
    r_xy=boxes[:,:2]+boxes[:,2:]/2
    return torch.cat([l_xy,r_xy],-1)

class PredTransform():
    def __init__(self, cfg,output_size):
        self.fea_map_sizes = cfg["feature_maps"]
        self.anchor_num = len(cfg["anchors"])
        self.steps = torch.Tensor(cfg["steps"]).float()
        self.scale = torch.Tensor([output_size[0]/cfg["input_size"],output_size[1]/cfg["input_size"],
                                   output_size[0]/cfg["input_size"],output_size[1]/cfg["input_size"]]).float()
        self.anchors_vec = self.build_anchor_vecs_(cfg["anchors"],cfg["steps"],cfg["anchor_scale"])
        self.cuda_device = cfg["cuda_device"]
        if not self.cuda_device == -1:
            self.anchors_vec = self.anchors_vec.cuda(self.cuda_device)
            self.steps = self.steps.cuda(self.cuda_device)
            self.scale = self.scale.cuda(self.cuda_device)
        self.batch_size = cfg["batch_size"]
        self.grids = self.build_grids_()
        self.num_class = cfg["num_class"]


    def __call__(self, pred_logit,threshold=0.5,nms=True):

        pred_boxs=[]
        pred_labels=[]
        pred_scores=[]
        for logit_,grid,step,anchor_vec in zip(pred_logit,self.grids,self.steps,self.anchors_vec):
            # anchor_vec = torch.Tensor(anchor/step).float().cuda()
            obj_logit = logit_[..., 4]
            obj = torch.sigmoid(obj_logit)
            pos_mask = obj>threshold
            if pos_mask.sum()<=0:
                continue
            pos_logit = logit_[pos_mask]
            pos_idx = grid[pos_mask]
            pred_xy = torch.sigmoid(pos_logit[...,:2])
            grid_xy = pos_idx[...,[1,0]]
            pred_xy = pred_xy +grid_xy
            pred_wh = torch.exp(pos_logit[...,2:4]).clamp(0,1e3)
            pred_wh = pred_wh*anchor_vec[pos_idx[...,2].long()]
            pred_box = torch.cat((pred_xy,pred_wh),-1)
            pred_box = pred_box*step*self.scale
            pred_boxs.append(pred_box)
            pred_score = torch.sigmoid(pos_logit[...,5:])
            if nms:
                pred_scores.append(pred_score)
            else:
                pred_label = pred_score.argmax(dim=-1)
                pred_labels.append(pred_label)
        if len(pred_boxs)>0:
            pred_boxs = torch.cat(pred_boxs,0)
            pred_boxs = xywh2LRpoint(pred_boxs)
            if nms:
                pred_scores = torch.cat(pred_scores,0)
            else:
                pred_labels = torch.cat(pred_labels, 0)
                return pred_boxs,pred_labels
        #nms
        res_boxes=[]
        res_labels=[]
        if len(pred_scores)<=0:
            return res_boxes,res_labels
        for cls_idx in range(self.num_class):
            ids = pred_scores[:,cls_idx]>threshold
            if ids.sum()<=0:
                continue
            pred_score = pred_scores[ids][...,cls_idx]
            pred_box = pred_boxs[ids]
            ids = nms_pytorch(boxes=pred_box.detach(),scores=pred_score.detach())
            res_boxes.append(pred_box[ids])
            res_labels.append(torch.ones_like(pred_score[ids])*cls_idx)

        res_boxes = torch.cat(res_boxes, 0)
        res_labels = torch.cat(res_labels, 0)



        return res_boxes,res_labels



    def build_anchor_vecs_(self,anchors,steps,scale):
        anchors = np.array(anchors)
        steps = np.array(steps)[:,np.newaxis,np.newaxis]
        steps = np.tile(steps,(1,anchors.shape[1],anchors.shape[2]))
        anchors_vec = anchors/steps*scale
        return torch.from_numpy(anchors_vec).float()

    def build_grids_(self):
        grids = []
        for map_size in self.fea_map_sizes:
            grid = torch.meshgrid(torch.arange(0, map_size),
                                  torch.arange(0, map_size),
                                  torch.arange(0, self.anchor_num)
                                  )
            grid = torch.cat([x.view(x.shape+torch.Size([1])) for x in grid], -1)
            grid = grid.expand(torch.Size([self.batch_size])+grid.shape).float().cuda()
            grids.append(grid)
        return grids
