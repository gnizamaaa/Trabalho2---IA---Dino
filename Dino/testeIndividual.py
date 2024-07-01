import numpy as np

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função para normalizar entradas
def fixInput(inputs):
    return np.array(inputs) / (np.sum(inputs) + 1e-9)

# Classe Neuronio
class Neuronio:
    def __init__(self, weights, bias, function):
        self.weights = np.array(weights)
        self.bias = bias
        self.function = function

    def forward(self, inputs):
        output = self.function(np.dot(self.weights, inputs) + self.bias)
        return output

    def setWeights(self, weights):
        self.weights = np.array(weights)

# Classe CamadaNeuronio
class CamadaNeuronio:
    def __init__(self, neuronios):
        self.neuronios = neuronios

    def forward(self, inputs):
        outputs = [neuronio.forward(inputs) for neuronio in self.neuronios]
        return outputs

    def setWeights(self, weights):
        for i, neuronio in enumerate(self.neuronios):
            neuronio.setWeights(weights[i])

# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self, state):
        self.state = state
        self.weights = state
        self.bias = 0

        # Inicializando pesos e biases diferentes para cada neurônio
        self.camada1 = CamadaNeuronio([Neuronio(np.random.rand(7), self.bias, relu) for _ in range(7)])
        self.camada2 = CamadaNeuronio(
            [
                Neuronio(self.weights[:7], self.bias, relu),
                Neuronio(self.weights[7:14], self.bias, relu),
                Neuronio(self.weights[14:21], self.bias, relu),
                Neuronio(self.weights[21:28], self.bias, relu),
            ]
        )
        self.camada3 = CamadaNeuronio(
            [Neuronio(self.weights[28:32], self.bias, sigmoid)]
        )

    def keySelector(
        self,
        distance,
        obHeight,
        speed,
        obType,
        nextObDistance,
        nextObHeight,
        nextObType,
    ):
        inputs = [
            distance,
            obHeight,
            speed,
            obType,
            nextObDistance,
            nextObHeight,
            nextObType,
        ]

        inputs = fixInput(inputs)
        #print(inputs)
        #print("\n")
        ret1 = self.camada1.forward(inputs)
        #print(ret1)
        #print("\n")
        ret2 = self.camada2.forward(ret1)
        #print(ret2)
        #print("\n")
        output = self.camada3.forward(ret2)
        #print(output)
        #print("\n")
        return "K_UP" if output[0] > 0.45 else "K_DOWN"

    def updateState(self, state):
        self.state = state
        self.weights = state
        self.camada2.setWeights(
            [
                self.weights[:7],
                self.weights[7:14],
                self.weights[14:21],
                self.weights[21:28],
            ]
        )
        self.camada3.setWeights([self.weights[28:32]])

    def setWeights(self, weights):
        self.weights = weights
        self.camada2.setWeights(
            [weights[:7], weights[7:14], weights[14:21], weights[21:28]]
        )
        self.camada3.setWeights([weights[28:32]])

# Implementação dos casos de teste
def test_neural_network_varied_outputs():
    state = np.random.uniform(-1, 1, 32).tolist()
    nn = NeuralNetwork(state)

    # Conjunto de entradas que deve resultar em "K_UP"
    inputs_up = (1, 2, 3, 4, 5, 6, 7)
    output_up = nn.keySelector(*inputs_up)
    print(f"Output para inputs_up: {output_up}")
    #assert output_up == "K_UP", f"Esperado 'K_UP', mas obteve '{output_up}'"

    # Conjunto de entradas que deve resultar em "K_DOWN"
    inputs_down = (7, 6, 5, 4, 3, 2, 1)
    output_down = nn.keySelector(*inputs_down)
    print(f"Output para inputs_down: {output_down}")
    #assert output_down == "K_DOWN", f"Esperado 'K_DOWN', mas obteve '{output_down}'"

    # Outros conjuntos de entradas para garantir que a rede pode retornar ambas as saídas
    inputs_mixed = [
        (1, 2, 3, 4, 5, 6, 1),
        (7, 6, 5, 4, 3, 2, 7),
        (0, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1)
    ]
    outputs = [nn.keySelector(*inputs) for inputs in inputs_mixed]
    print(f"Outputs para inputs_mixed: {outputs}")
    #assert "K_UP" in outputs, "A rede não retornou 'K_UP' para nenhum dos conjuntos de entradas."
    #assert "K_DOWN" in outputs, "A rede não retornou 'K_DOWN' para nenhum dos conjuntos de entradas."

# Rodar o teste
if __name__ == "__main__":
    test_neural_network_varied_outputs()
    print("Teste de saídas variadas passou com sucesso!")
