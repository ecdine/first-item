from loss import KeypointLoss
from transformer import Transformer
from transformer import MLP
from position_encoding import PositionEmbeddingSine
from wipose import WifiPoseDataset
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
import torch
from matcher import HungarianMatcher, Match
import os

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(60)
        self.pos_embed = PositionEmbeddingSine()  
        self.query_embed = nn.Embedding(100, 60)
        self.transformer = Transformer() 
        self.linear_class = nn.Linear(in_features=60, out_features=2)
        self.MLP = MLP(60, 60, 34, 3)
        self.matcher = HungarianMatcher(cost_class=2.0, cost_oks=7.0, cost_kpt=70.0)

    def forward(self, img):
        mask = torch.ones(img.size(0), 1350)
        img = self.batch_norm(img)
        pos = self.pos_embed(img)##bs,60,27,50
        query = self.query_embed.weight
        output, memory = self.transformer(img, mask, query, pos)
        output = output.squeeze(0)
        outputs = {}
        outputs["pred_logits"] = self.linear_class(output)
        outputs["pred_keypoints"] = self.MLP(output)
        return outputs
    
if __name__ == "__main__":
    batchsize = 6
    dataset = WifiPoseDataset(dataset_root='/megadisk/fanghengyu/XRF55/opera/datasets/train', mode='train')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn= WifiPoseDataset.custom_collate_fn)
    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_calculator = KeypointLoss(weight_ce=2.0, weight_oks=4.0, weight_l1=60.0, alpha=0.8, gamma=3.0)
    checkpoint_path = '/megadisk/fanghengyu/WiSPPN/path/p100.pth'  
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print("成功加载模型")
    else:
        print("开始新的训练")
    model.train()
    epochs = 1000
    for epoch in range(epochs):
        for img, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(img)
            with torch.no_grad():
                indices = model.matcher(outputs, targets) 
            ##print(indices)
            src_logits, target_classes, pred, tgt_keypoints, tgt_area = Match.keypoint_match(outputs, targets, indices)
            loss = loss_calculator.compute_loss(src_logits, target_classes, pred, tgt_keypoints, tgt_area)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}" )
            if (epoch+1) % 200 == 0:
                Pth = f'path/q{epoch+1}.pth'
                torch.save(model.state_dict(), Pth)
