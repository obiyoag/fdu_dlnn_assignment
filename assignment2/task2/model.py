from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOSHead, FCOS_ResNet50_FPN_Weights


def create_faster_rcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, fixed_size=[512, 512])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


def create_fcos(num_classes):
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT, fixed_size=[512, 512])
    model.head = FCOSHead(model.backbone.out_channels, model.anchor_generator.num_anchors_per_location()[0], num_classes)
    return model
