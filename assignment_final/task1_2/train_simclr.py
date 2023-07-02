import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from resnet import resnet18
from cifar_trainer import Trainer


class ContrastiveLearningViewGenerator:
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class GaussianBlur:
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img).squeeze()

        img = self.tensor_to_pil(img)

        return img


class PreTrainer(Trainer):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * 32)),
                                              transforms.ToTensor()])

        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=ContrastiveLearningViewGenerator(data_transforms))
        self.train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
        self.model = resnet18(supervised=False).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(self.epochs * len(self.train_loader)))

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(256) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / 0.07
        return logits, labels
    

    def train_one_epoch(self, epoch):
        train_loss = 0.0
        train_acc = 0.0

        for images, _ in self.train_loader:
            images = torch.cat(images, dim=0).to(self.device)

            features = self.model(images)
            logits, labels = self.info_nce_loss(features)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += self.accuracy(logits, labels)[0].item()

        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)
        self.scheduler.step()

        print(f"Epoch [{epoch+1}/{self.epochs}]\ttrain_loss: {train_loss:.4f}\ttrain_acc: {train_acc:.2f}%")
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)


    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)

        self.writer.close()
        torch.save(self.model.state_dict(), f'{self.log_dir}/{self.exp_name}.pth')


if __name__ == '__main__':
    trainer = PreTrainer(exp_name='simclr')
    trainer.train()
