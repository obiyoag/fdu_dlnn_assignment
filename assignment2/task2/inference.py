import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from PIL import Image
from torchvision.transforms import ToTensor

from utils import voc_classes
from data import get_loaders
from model import create_faster_rcnn, create_fcos


@torch.no_grad()
def show_proposals(model):
    _, test_loader = get_loaders()
    images, targets = next(iter(test_loader))
    images = [image.cuda() for image in images]
    targets = [{k: v.cuda() for k, v in target.items()} for target in targets]

    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features, targets)


    for idx in range(3):
        image, proposal = images.tensors[idx].cpu().numpy(), proposals[idx].cpu().numpy()[:10]

        x_min = proposal[:, 0]
        y_min = proposal[:, 1]
        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]

        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image.transpose(1, 2, 0))
        plt.axis('off')

        for i in range(len(proposal)):
            rect = patches.Rectangle((x_min[i], y_min[i]), width[i], height[i], linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

        plt.show()
        plt.savefig(f'imgs/proposals/{idx}.png', bbox_inches='tight')
        plt.close()


@torch.no_grad()
def show_result(model):
    images = []
    image_paths = glob('imgs/ood_images/*')
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = ToTensor()(image).cuda()
        images.append(image)
    
    outputs = model(images)
    for idx, (image, output) in enumerate(zip(images, outputs)):

        image = image.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        boxes, labels, scores = boxes[scores >= 0.75], labels[scores >= 0.75], scores[scores >= 0.75]

        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        plt.imshow(image.transpose(1, 2, 0))
        plt.axis('off')

        for i in range(len(labels)):
            rect = patches.Rectangle((x_min[i], y_min[i]), width[i], height[i], linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.gca().text(x_min[i] + 3, y_min[i] - 3, f'label: {voc_classes[labels[i]]}, score: {scores[i]:.3f}', fontweight='bold')

        plt.show()
        plt.savefig(f'imgs/results/{idx}.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--show_proposals', action='store_true')
    parser.add_argument('--show_results', action='store_true')
    args = parser.parse_args()

    if args.model_name == 'faster_rcnn':
        model = create_faster_rcnn(num_classes=20)
    elif args.model_name == 'fcos':
        model = create_fcos(num_classes=20)

    checkpoint = torch.load(f'./logs/{args.model_name}/{args.model_name}.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval().cuda()


    if args.show_proposals and args.model_name == 'faster_rcnn':
        show_proposals(model)
    
    if args.show_results:
        show_result(model)
