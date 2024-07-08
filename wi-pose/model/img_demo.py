import torch
from train import TransformerModel 
from wipose import CSI
import numpy as np
import cv2
import time
def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
 
 
def non_max_suppression(boxes, scores, iou_threshold=0.5):

    sorted_indices = np.argsort(scores)[::-1]
 
    keep_boxes = []
    while sorted_indices.size > 0:
        idx = sorted_indices[0]
        keep_boxes.append(idx)
 
        ious = np.array([intersection_over_union(boxes[idx], boxes[m]) for m in sorted_indices[1:]])
 
        remove_indices = np.where(ious > iou_threshold)[0] + 1 
        sorted_indices = np.delete(sorted_indices, remove_indices)
        sorted_indices = np.delete(sorted_indices, 0) 
    return keep_boxes

def load_model_and_run_images():
    csi_path = '/megadisk/fanghengyu/XRF55/opera/datasets/train/csi'
    MODEL = CSI()
    results = MODEL.get_csi(csi_path)
    model = TransformerModel().to(device='cuda')
    model.load_state_dict(torch.load('/megadisk/fanghengyu/WiSPPN/path/q400.pth', map_location='cuda'))
    model.train()
    i = 1
    for tensor in results:
        assert isinstance(tensor, torch.Tensor), "The element is not a PyTorch tensor."
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device='cuda')
        outputs = model(tensor)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        probas = probas.detach().cpu().numpy()
        probas = probas.reshape(100)
        keypoints = outputs['pred_keypoints']
        keypoints = keypoints.reshape(-1, 17, 2)
        keypoints[:, :, 0] = keypoints[:, :, 0] * 640
        keypoints[:, :, 1] = keypoints[:, :, 1] * 480
        keypoints = keypoints.detach().cpu().numpy()
        print(keypoints)
        boxes = np.zeros((keypoints.shape[0], 4))
        for n, kp in enumerate(keypoints):
            min_x = np.min(kp[:, 0])
            max_x = np.max(kp[:, 0])
            min_y = np.min(kp[:, 1])
            max_y = np.max(kp[:, 1])
            boxes[n] = [min_x, min_y , max_x, max_y]
        iou_thresh = 0.2
        indices_to_keep = non_max_suppression(boxes, probas, iou_threshold=iou_thresh)
        keypoints = keypoints[indices_to_keep, :, :]
        img_size = (640, 1000) 
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
    load_model_and_run_images()
