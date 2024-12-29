import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=1, keepdims=True)

#для бинарной классификации
def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# для многоклассовой классификации
def mnogoclass_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))

def neural_network(input_data, weights_hidden, bias_hidden, weights_output, bias_output, activation='relu'):
    hidden_layer = relu(np.dot(input_data, weights_hidden) + bias_hidden)
    if activation == 'sigmoid':
        output = sigmoid(np.dot(hidden_layer, weights_output) + bias_output)
    elif activation == 'softmax':
        output = softmax(np.dot(hidden_layer, weights_output) + bias_output)
    return output

X = np.array([[0, 1, 0],
              [1, 0, 1],
              [1, 1, 1]])

# Бинарная
y_binary = np.array([[0],
                  [1],
                  [1]])

# Многоклассовая
y_multiclass = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

weights_hidden = np.random.rand(3, 4)
bias_hidden = np.random.rand(4)
weights_output = np.random.rand(4, 1)
bias_output = np.random.rand(1)

print("Бинарная классификация:")
for i in range(len(X)):
    output = neural_network(X[i].reshape(1, -1), weights_hidden, bias_hidden, weights_output, bias_output, activation='sigmoid')
    loss = binary_crossentropy(y_binary[i], output)
    print(f"Экземпляр {i+1}, Потери: {loss}")

weights_output_multiclass = np.random.rand(4, 3)
bias_output_multiclass = np.random.rand(3)
print("\nМногоклассовая классификация:")
for i in range(len(X)):
    output_multiclass = neural_network(X[i].reshape(1, -1), weights_hidden, bias_hidden, weights_output_multiclass, bias_output_multiclass, activation='softmax')
    loss_multiclass = mnogoclass_crossentropy(y_multiclass[i], output_multiclass)
    print(f"Экземпляр {i+1}, Потери: {loss_multiclass}")