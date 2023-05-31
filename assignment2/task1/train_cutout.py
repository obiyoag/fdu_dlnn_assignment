import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from cifar100_trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size()[1:]
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, size=(1,)).item()
        x = torch.randint(w, size=(1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        img = img * mask.unsqueeze(0)
        return img


class Cutout_Trainer(Trainer):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar100_mean, self.cifar100_std),
            Cutout(16),
        ])

        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    
    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        visualized = False

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)


            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if epoch == 0 and not visualized:
                visualized = True
                for i in range(3):
                    image = images[i].permute(1, 2, 0).cpu().numpy()
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
    trainer = Cutout_Trainer(exp_name='cutout')
    trainer.train()
