import torch
from train import TransformerModel 
from wipose import CSI
import numpy as np
import cv2
def load_model_and_run_images():
    csi_path = '/megadisk/fanghengyu/XRF55/opera/datasets/train/csi'
    MODEL = CSI()
    results = MODEL.get_csi(csi_path)
    model = TransformerModel()
    model.load_state_dict(torch.load('/megadisk/fanghengyu/WiSPPN/path/q400.pth'))
    model.train() 
    i = 1
    for tensor in results:
        assert isinstance(tensor, torch.Tensor), "The element is not a PyTorch tensor."
        tensor = tensor.unsqueeze(0)
        outputs = model(tensor)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        print(probas)
        keep = probas.max(-1).values > 0.2
        keypoints = outputs['pred_keypoints'][0, keep]
        keypoints = keypoints.reshape(-1, 17, 2)
        keypoints[:, :, 0] = keypoints[:, :, 0] * 640
        keypoints[:, :, 1] = keypoints[:, :, 1] * 480
        img_size = (640, 480) 
        background = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
        limbs = [(0, 1), (1, 3), (0, 2), (2, 4),  
         (5, 7), (7, 9), (6, 8), (8, 10), 
         (5, 6), (11, 12), (5, 11), (6, 12), 
         (11, 13), (13, 15), (12, 14), (14, 16)]
        for person in keypoints:
            for point in person:
                x, y = int(point[0]), int(point[1])
                cv2.circle(background, (x, y), 3, (0, 0, 255), -1) 
            for limb in limbs:
                start_point = person[limb[0]]
                end_point = person[limb[1]]
                cv2.line(background, (int(start_point[0]), int(start_point[1])), 
                        (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2) 
        ##cv2.imshow('Keypoints Visualization', background)
        ##cv2.waitKey(0) 由于在SHH上不好调用xcb
        cv2.imwrite(f'outputq{i}.jpg', background)
        cv2.destroyAllWindows()
        i += 1          
if __name__ == "__main__":
    batch_size = 6
    dataset_root = '/megadisk/fanghengyu/XRF55/opera/datasets/train'
    load_model_and_run_images()
