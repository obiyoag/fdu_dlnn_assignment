import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from resnet import resnet18


class Trainer:
    def __init__(self, exp_name):

        self.exp_name = exp_name

        torch.manual_seed(42)

        self.cifar100_mean, self.cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar100_mean, self.cifar100_std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cifar100_mean, self.cifar100_std)
        ])

        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        self.model = resnet18()

        self.epochs = 200
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.log_dir = f'./logs_new/{exp_name}'
        self.writer = SummaryWriter(self.log_dir)


    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

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

        train_loss /= len(self.train_loader)
        train_acc = 100. * correct / total

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttrain_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.2f}%")
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)


    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttest_loss: {test_loss:.4f}\ttest_acc: {test_acc:.2f}%")
        self.writer.add_scalar('test_loss', test_loss, epoch)
        self.writer.add_scalar('test_acc', test_acc, epoch)


    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.scheduler.step()
            if epoch % 5 == 0:
                self.evaluate(epoch)
        self.writer.close()
        torch.save(self.model.state_dict(), f'{self.log_dir}/{self.exp_name}.pth')

