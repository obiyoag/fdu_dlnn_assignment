import torch
from podm.metrics import MetricPerClass
from podm.box import Box, intersection_over_union
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, BoundingBox


voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class AnnotationTransform:

    def __init__(self):
        self.class_to_ind = dict(zip(voc_classes, range(len(voc_classes))))

    def __call__(self, targets):
        height = int(targets['annotation']['size']['height'])
        width = int(targets['annotation']['size']['width'])

        labels, boxes = [], []
        obj_anns = targets['annotation']['object']
        for obj_ann in obj_anns:
            labels.append(self.class_to_ind[obj_ann['name']])
            # xmin = (int(obj_ann['bndbox']['xmin']) - 1) / width
            # ymin = (int(obj_ann['bndbox']['ymin']) - 1) / height
            # xmax = (int(obj_ann['bndbox']['xmax']) - 1) / width
            # ymax = (int(obj_ann['bndbox']['ymax']) - 1) / height
            xmin = (int(obj_ann['bndbox']['xmin']) - 1)
            ymin = (int(obj_ann['bndbox']['ymin']) - 1)
            xmax = (int(obj_ann['bndbox']['xmax']) - 1)
            ymax = (int(obj_ann['bndbox']['ymax']) - 1)
            boxes.append([xmin, ymin, xmax, ymax])

        return {'labels': torch.tensor(labels), 'boxes': torch.tensor(boxes)}


def calculate_iou(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    iou = torch.diag(inter / union)

    enclose_mins = torch.min(box_a[:, :2], box_b[:, :2])
    enclose_maxes = torch.max(box_a[:, 2:], box_b[:, 2:])
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(enclose_maxes))

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - torch.diag(union)) / enclose_area
    
    return iou, giou


def get_mIOU(pre_boxes,target_boxes,iou_threshold):
    mIOU_list_each = []
    matched_target = []
    for k in range(len(pre_boxes)):
        for j in range(len(target_boxes)):
            box1 = Box.of_box(pre_boxes[k][0], pre_boxes[k][1], pre_boxes[k][2], pre_boxes[k][3])
            box2 = Box.of_box(target_boxes[j][0], target_boxes[j][1], target_boxes[j][2], target_boxes[j][3])
            IOU = intersection_over_union(box1, box2)
            if IOU >= iou_threshold and j not in matched_target:
                matched_target.append(j)
                mIOU_list_each.append(IOU)
    mIOU = sum(mIOU_list_each) / (len(mIOU_list_each)+1e-5)
    return mIOU


def get_mAP(pre_class,pre_boxes,pre_scores,target_class,target_boxes,iou_threshold):
    pre_box = []
    for i in range(len(pre_class)):
        bb = BoundingBox.of_bbox(None,category = pre_class[i],xtl = pre_boxes[i][0],ytl = pre_boxes[i][1],xbr = pre_boxes[i][2], ybr = pre_boxes[i][3],score = pre_scores[i])
        pre_box.append(bb)
    
    target_box = []
    for i in range(len(target_class)):
        bb = BoundingBox.of_bbox(None,category = target_class[i],xtl = target_boxes[i][0],ytl = target_boxes[i][1],xbr = target_boxes[i][2], ybr = target_boxes[i][3])
        target_box.append(bb)
    results = get_pascal_voc_metrics(target_box, pre_box, iou_threshold)

    tps_list = []
    for cls, metric in results.items():
        tps_list.append(metric.tp)
    tp = sum(tps_list)/len(tps_list)
    acc = tp / (len(pre_class)+1e-5)
    mAP = MetricPerClass.mAP(results)
    return mAP,acc
