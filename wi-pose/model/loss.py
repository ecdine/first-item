import torch
import torch.nn.functional as F
import numpy as np

class KeypointLoss:
    def __init__(self, weight_ce=1, weight_oks=1, weight_l1=1, alpha=0.75, gamma=2.0):
        self.weight_ce = weight_ce
        self.weight_oks = weight_oks
        self.weight_l1 = weight_l1
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs.transpose(1, 2), targets, reduction='none')  
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  
        return focal_loss.mean()  

    def oks_loss(self, kpt_preds, kpt_gts, area):
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
            1.07, .87, .87, .89, .89
        ], dtype=np.float32) / 10.0
        variances = (sigmas * 2)**2

        kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
        kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)
        squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
            (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
        kpt_valids = torch.ones(len(kpt_gts), 17)
        squared_distance0 = squared_distance / (area[:, None] * variances[None, :] * 2)
        squared_distance1 = torch.exp(-squared_distance0)
        squared_distance1 = squared_distance1 * kpt_valids
        oks = squared_distance1.sum(dim=1) / kpt_valids.sum(dim=1)
        epsilon = 1e-8
        loss = 1-torch.log(oks+epsilon).mean()
        return loss

    def l1_loss(self, kpt_preds, kpt_gts):
        return F.l1_loss(kpt_preds, kpt_gts, reduction='mean')

    def compute_loss(self, src_logits, target_classes, pred, tgt_keypoints, tgt_area):
        loss_ce = self.focal_loss(src_logits, target_classes)  # 使用Focal Loss计算loss_ce
        loss_oks = self.oks_loss(pred, tgt_keypoints, tgt_area)
        loss_l1 = self.l1_loss(pred, tgt_keypoints)
        total_loss = (loss_ce * self.weight_ce) + (loss_oks * self.weight_oks) + (loss_l1 * self.weight_l1)
        print(f"cls_loss:{loss_ce.item()}"+f"oks_loss:{loss_oks.item()}"+f"L1_loss:{loss_l1.item()}")
        return total_loss