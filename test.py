import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from layers import Linear, ReLU, SoftmaxWithLoss


class Tester:
    def __init__(self):
        dataset = MNIST(root='./data', train=False, download=True)
        self.images, self.labels = np.array(dataset.data).reshape(10000, -1), np.array(dataset.targets)

        self.params = np.load('params/lr1e-05_hd512_wd0.0001.npy', allow_pickle=True).item()

        self.linear1 = Linear(self.params['weight1'], self.params['bias1'], lr=None, weight_decay=None)
        self.relu = ReLU()
        self.linear2 = Linear(self.params['weight2'], self.params['bias2'], lr=None, weight_decay=None)
        self.loss = SoftmaxWithLoss()

    def test(self):
        out = self.linear1.forward(self.images)
        out = self.relu.forward(out)
        out = self.linear2.forward(out)
        self.loss.cal_loss(out, self.labels)

        acc = 0
        for j in range(self.images.shape[0]):
            if np.argmax(self.loss.softmax[j]) == self.labels[j]:
                acc += 1
        acc /= self.images.shape[0]

        print(f"test acc: {acc}")

    def visualize_weight(self):
        weight1 = self.params['weight1']
        weight2 = self.params['weight2']

        fig, axes = plt.subplots(2)
        axes[0].imshow(weight1.transpose(), cmap='gray')
        axes[0].set_title('layer1')
        axes[1].imshow(weight2.transpose(), cmap='gray')
        axes[1].set_title('layer2')

        plt.savefig('visualize/weight.png')


if __name__ == '__main__':
    tester = Tester()
    tester.test()
    tester.visualize_weight()
