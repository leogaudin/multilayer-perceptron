class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
