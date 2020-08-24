from src.activation_functions import Activation, Linear


class Layer:
    def __init__(self, neuron_count: int, activation_func: Activation, next_layer=None, weight=None):
        self.n_neurons = neuron_count
        self.activation_func = activation_func
        self.pre_activation = None
        self.activation = None
        self.next_layer = next_layer
        self.target = None
        self.weight = weight  # weight matrix connecting to next layer

    def __str__(self):
        return f"#Neurons: {self.n_neurons}"


class InputLayer(Layer):
    def __init__(self, neuron_count: int, *args, **kwargs):
        super().__init__(neuron_count, Linear())
