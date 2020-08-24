import numpy as np
from src.neural_net.layers import Layer


class Weight:
    def __init__(self, prev_layer: Layer, next_layer: Layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.dim_out, self.dim_in = next_layer.n_neurons, prev_layer.n_neurons
        self.W = np.random.randn(self.dim_out, self.dim_in) * 0.1
        self.Omega = np.random.randn(self.dim_in, self.dim_out) * 0.1
        self.bias = np.random.randn(self.dim_out, 1) * 0.1  # TODO support bias
        self.weight_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)
        self.activation_func = self.next_layer.activation_func
        self.n_accs = 0
        self.lr = None

    def forward(self, x):
        self.prev_layer.activation = x
        self.next_layer.activation = self.encode(x)
        if x.shape[1] != 1: # e.g. for validation pass, where there won't be a backward pas
            return self.next_layer.activation
        sigma_x = self.activation_func(x)
        self.prev_layer.pre_activation = sigma_x / np.linalg.norm(sigma_x)**2

        act2 = self.activation_func(self.next_layer.activation).T
        self.Omega += self.lr * (x - self.decode(self.next_layer.activation)) @ \
                      act2 / np.linalg.norm(act2)**2

        return self.next_layer.activation

    def init_target(self, target):
        self.next_layer.target = target
        self.prev_layer.target = self.decode(target)
        return self.prev_layer.target

    def decode(self, x):
        return self.Omega @ self.activation_func(x)

    def encode(self, x):
        return self.W @ self.activation_func(x)

    def target_prop(self):
        self.prev_layer.target += self.decode(self.next_layer.target) - \
                                  self.decode(self.encode(self.prev_layer.target))
        return self.prev_layer.target

    def backward(self, times=1.0):
        diff = self.next_layer.target - self.next_layer.activation
        self.weight_acc += times * diff @ self.prev_layer.pre_activation.T / np.linalg.norm(diff) ** 2
        self.n_accs += 1

    def update(self):
        self.W = self.W + self.weight_acc / self.n_accs
        # self.bias = self.bias + self.bias_acc / self.n_accs
        self.weight_acc = np.zeros(self.W.shape)
        # self.bias_acc = np.zeros(self.bias.shape)
        self.n_accs = 0
