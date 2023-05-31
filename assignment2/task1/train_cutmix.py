import torch
import numpy as np
import matplotlib.pyplot as plt

from cifar100_trainer import Trainer


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

    

class Cutmix_Trainer(Trainer):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        self.beta = 1.0
        self.cutmix_prob = 0.5

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        visualized = False

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            r = np.random.rand(1)
            if r < self.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_index = torch.randperm(images.size(0)).to(self.device)
                target_a, target_b = labels, labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)

                if epoch == 0 and not visualized:
                    visualized = True
                    for i in range(3):
                        image = images[i].permute(1, 2, 0).cpu().numpy()
                        image = (image - image.min()) / (image.max() - image.min())
                        plt.imshow(image)
                        plt.axis('off')
                        plt.savefig(f'{self.log_dir}/{self.exp_name}_{i}', bbox_inches='tight')

            else:
                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(self.train_loader)
        train_acc = 100. * correct / total

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttrain_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.2f}%")
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)


if __name__ == '__main__':
    trainer = Cutmix_Trainer(exp_name='cutmix')
    trainer.train()
