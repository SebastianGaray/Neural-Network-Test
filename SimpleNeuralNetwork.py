import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    # Funcion que actualiza el valor de y
    # y = sigmoid(W2*sigmoid(W1 + b1) + b2)
    # Asumiendo sesgo = 0
    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
   
    # Funcion que acualiza los pesos de la red
    def backpropagation(self):
        # Se deriva la funcion que calcula el error de la red, con respecto a los pesos
        dWeights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        dWeights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Se actualizan los pesos
        self.weights1 += dWeights1
        self.weights2 += dWeights2



X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)

for i in range(1500):
    nn.feedForward()
    nn.backpropagation()

print(nn.output)

