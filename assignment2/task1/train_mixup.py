import torch
import numpy as np
import matplotlib.pyplot as plt

from cifar100_trainer import Trainer


class Mixup_Trainer(Trainer):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        self.alpha= 1.0

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        visualized = False

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(images.size(0)).to(self.device)
            mixed_images = lam * images + (1 - lam) * images[rand_index, :]
            target_a, target_b = labels, labels[rand_index]

            outputs = self.model(mixed_images)
            loss = lam * self.criterion(outputs, target_a) + (1 - lam) * self.criterion(outputs, target_b)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(target_a).cpu().sum().float() + (1 - lam) * predicted.eq(target_b).cpu().sum().float())

            if epoch == 0 and not visualized:
                    visualized = True
                    for i in range(3):
                        image = mixed_images[i].permute(1, 2, 0).cpu().numpy()
                        image = (image - image.min()) / (image.max() - image.min())
                        plt.imshow(image)
                        plt.axis('off')
                        plt.savefig(f'{self.log_dir}/{self.exp_name}_{i}', bbox_inches='tight')

        train_loss /= len(self.train_loader)
        train_acc = 100. * correct / total

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttrain_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.2f}%")
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)


if __name__ == '__main__':
    trainer = Mixup_Trainer(exp_name='mixup')
    trainer.train()
