import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST

from layers import Linear, ReLU, SoftmaxWithLoss


class Trainer:
    def __init__(self, lr=1e-4, hidden_dim=256, weight_decay=4e-4):
        dataset = MNIST(root='./data', train=True, download=True)
        images, labels = np.array(dataset.data), np.array(dataset.targets)
        self.train_images, self.train_labels = images[:50000], labels[:50000]
        self.val_images, self.val_labels = images[50000:].reshape(10000, -1), labels[50000:]

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        self.setting = f'lr{self.lr}_hd{self.hidden_dim}_wd{self.weight_decay}'

        self.iter_num = 0
        self.batch_size = 10
        self.epoch = 10

        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.best_acc = 0

        self.total_steps = (self.train_images.shape[0] // self.batch_size) * self.epoch

        self.params = {
            'weight1': np.random.randn(28*28, hidden_dim) * np.sqrt(1 / hidden_dim),
            'bias1': np.random.randn(hidden_dim) * np.sqrt(1 / hidden_dim),
            'weight2': np.random.randn(hidden_dim, 10) * np.sqrt(1 / 10),
            'bias2': np.random.randn(10) * np.sqrt(1 / 10),
        }

        self.linear1 = Linear(self.params['weight1'], self.params['bias1'], lr, weight_decay)
        self.relu = ReLU()
        self.linear2 = Linear(self.params['weight2'], self.params['bias2'], lr, weight_decay)
        self.loss = SoftmaxWithLoss()

    def sample_batch_data(self, iter_idx):
        image = self.train_images[iter_idx*self.batch_size: (iter_idx+1)*self.batch_size].reshape(self.batch_size, -1)
        label = self.train_labels[iter_idx*self.batch_size: (iter_idx+1)*self.batch_size]
        return image, label

    def lr_schedule(self):
        progress = self.iter_num / self.total_steps
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) * self.lr

    def validate(self):
        out = self.linear1.forward(self.val_images)
        out = self.relu.forward(out)
        out = self.linear2.forward(out)
        loss = self.loss.cal_loss(out, self.val_labels)

        acc = 0
        for j in range(self.val_images.shape[0]):
            if np.argmax(self.loss.softmax[j]) == self.val_labels[j]:
                acc += 1
        acc /= self.val_images.shape[0]

        print(f"validation acc: {acc}")
        self.val_loss.append(loss)
        self.val_acc.append(acc)

        if acc > self.best_acc:
            self.best_acc = acc
            np.save(f'params/{self.setting}.npy', self.params)

    def visualize(self):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(list(range(self.total_steps)), self.train_loss)
        axes[0].set_title('trian_loss')
        axes[1].plot(list(range(self.epoch)), self.val_loss)
        axes[1].set_title('val_loss')
        axes[2].plot(list(range(self.epoch)), self.val_acc)
        axes[2].set_title('val_acc')

        plt.savefig(f'visualize/{self.setting}.png')
        
    def train(self):
        for epoch_idx in range(self.epoch):
            for iter_idx in range(self.train_images.shape[0] // self.batch_size):

                image, label = self.sample_batch_data(iter_idx)

                out = self.linear1.forward(image)
                out = self.relu.forward(out)
                out = self.linear2.forward(out)
                batch_loss = self.loss.cal_loss(out, label)
                self.train_loss.append(batch_loss)

                batch_acc = 0
                for j in range(self.batch_size):
                    if np.argmax(self.loss.softmax[j]) == label[j]:
                        batch_acc += 1

                self.linear1.backward(self.relu.backward(self.linear2.backward(self.loss.backward())))

                self.linear1.update(lr=self.lr_schedule())
                self.linear2.update(lr=self.lr_schedule())

                if iter_idx % 100 == 0:
                    print(f"epoch: {epoch_idx}  batch_acc: {batch_acc/self.batch_size}  batch_loss: {batch_loss}")

                batch_acc = 0
                self.iter_num += 1

            self.validate()
        self.visualize()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
