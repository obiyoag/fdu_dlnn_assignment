import numpy as np


class Linear:
    def __init__(self, weight, bias, lr, weight_decay):
        self.weight = weight
        self.bias = bias
        self.lr = lr
        self.weight_decay = weight_decay

        self.w_grad = np.zeros_like(self.weight)
        self.b_grad = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.weight) + self.bias

    def backward(self, delta):
        for i in range(delta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            delta_i = delta[i][:, np.newaxis].T
            self.w_grad += np.dot(col_x, delta_i)
            self.b_grad += delta_i.reshape(self.bias.shape)

        next_delta = np.dot(delta, self.weight.T)

        return next_delta

    def update(self, lr=None):
        if lr is None:
            lr = self.lr
        # L2 regularization
        self.weight *= (1 - self.weight_decay)
        self.bias *= (1 - self.weight_decay)
        # update weights
        self.weight -= lr * self.w_grad
        self.bias -= lr * self.b_grad
        # set zero gradients
        self.w_grad = np.zeros_like(self.weight)
        self.b_grad = np.zeros_like(self.bias)


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta


class SoftmaxWithLoss:
    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(prediction.shape[0]):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        return self.loss / prediction.shape[0]

    def predict(self, prediction):
        exp_prediction = np.zeros_like(prediction)
        self.softmax = np.zeros_like(prediction)
        for i in range(prediction.shape[0]):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])
        return self.softmax

    def backward(self):
        self.delta = self.softmax.copy()
        for i in range(self.delta.shape[0]):
            self.delta[i, self.label[i]] -= 1
        return self.delta
