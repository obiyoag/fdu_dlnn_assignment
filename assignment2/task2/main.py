import torch
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import get_loaders
from utils import get_mAP, get_mIOU
from model import create_faster_rcnn, create_fcos


class Trainer:
    def __init__(self, model_name):

        self.model_name = model_name

        torch.manual_seed(42)

        self.train_loader, self.test_loader = get_loaders()

        if self.model_name == 'faster_rcnn':
            self.model = create_faster_rcnn(num_classes=20).cuda()
        elif self.model_name == 'fcos':
            self.model = create_fcos(num_classes=20).cuda()

        self.epochs = 50
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.log_dir = f'./logs/{model_name}'
        self.writer = SummaryWriter(self.log_dir)


    def train_one_epoch(self, epoch):
        train_loss = 0.0
        for images, targets in self.train_loader:

            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in target.items()} for target in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step() 
            train_loss += losses.item()

        train_loss /= len(self.train_loader)

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttrain_loss: {train_loss:.4f}")
        self.writer.add_scalar('train_loss', train_loss, epoch)


    @torch.no_grad()
    def evaluate(self, epoch):

        test_loss = 0.0
        mAP_list = []
        mIOU_list = []
        acc_list = []

        for images, targets in self.test_loader:

            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in target.items()} for target in targets]

            loss_dict = self.model(images, targets)
            self.model.eval()
            detections = self.model(images)
            self.model.train()
            
            for i in range(len(detections)):
                target_class = targets[i]['labels'].cpu().numpy().tolist()
                target_boxes = targets[i]['boxes'].cpu().numpy().tolist()
                
                pre_scores = detections[i]['scores'].cpu().detach().numpy()
                pre_boxes = detections[i]['boxes'].cpu().detach().numpy().tolist()
                pre_class = detections[i]['labels'].cpu().detach().numpy().tolist()
                mAP_tmp, acc_tmp = get_mAP(pre_class, pre_boxes, pre_scores, target_class, target_boxes, iou_threshold=0.5)

                mAP_list.append(mAP_tmp)
                acc_list.append(acc_tmp)
                mIOU_list.append(get_mIOU(pre_boxes, target_boxes, iou_threshold=0.5))

            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()

        test_loss /= len(self.test_loader)
        mAP = sum(mAP_list) / len(mAP_list)
        acc = sum(acc_list) / len(acc_list)
        mIOU = sum(mIOU_list) / len(mIOU_list)

        print(f"Epoch [{epoch+1}/{self.epochs}]\tmAP: {mAP:.4f}\tacc: {acc:.4f}\tmIOU: {mIOU:.4f}")
        self.writer.add_scalar('test_loss', test_loss, epoch)
        self.writer.add_scalar('mAP', mAP, epoch)
        self.writer.add_scalar('acc', acc, epoch)
        self.writer.add_scalar('mIOU', mIOU, epoch)


    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.scheduler.step()
            if epoch % 5 == 0:
                self.evaluate(epoch)
        self.writer.close()
        torch.save(self.model.state_dict(), f'{self.log_dir}/{self.model_name}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    trainer = Trainer(model_name=args.model_name)
    trainer.train()
