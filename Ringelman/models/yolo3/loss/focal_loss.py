import torch
# class FocalLoss():
#     def __init__(self,alpha=0.25,gamma=1.5,reduction="mean"):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.BCELoss = torch.nn.BCELoss(reduction='none')
#
#     def __call__(self, input_data,target_data):
#         # target/input shape:[img_idx,gridy,gridx,obj_cls]
#         input = input_data.sigmoid()
#         loss = self.BCELoss(input,target_data)
#
#         focal_weight = ((self.alpha)*((1-input)**self.gamma)*target_data) + ((1-self.alpha)*(input**self.gamma)*(1-target_data))
#         loss = focal_weight*loss
#         if self.reduction == "mean":
#             return loss.mean()
#         else:
#             return loss.sum()


class FocalLoss(torch.nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, alpha=0.25,gamma=1.5,reduction="mean",pos_weight=None):
        super(FocalLoss, self).__init__()
        self.loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight) # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss