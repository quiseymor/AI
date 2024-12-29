import numpy as np
import tensorflow as tf

def relu(x):
    return tf.maximum(0.0, x)

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Функция потерь
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

# Параметры нс
input_size = 3      # Количество входов
hidden_size = 4     # Количество нейронов в скрытом слое
output_size_binary = 1  # Нейрон выходного слоя для бинарной классификации

np.random.seed(42)
weights_input_hidden = tf.Variable(tf.random.uniform((input_size, hidden_size)), dtype=tf.float32)
weights_hidden_output_binary = tf.Variable(tf.random.uniform((hidden_size, output_size_binary)), dtype=tf.float32)

def forward_pass_binary(X):
    hidden_layer_input = tf.matmul(X, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = tf.matmul(hidden_layer_output, weights_hidden_output_binary)
    predicted_output = sigmoid(output_layer_input)

    return hidden_layer_input, hidden_layer_output, predicted_output

def backward_pass(X, y, learning_rate=0.1):
    with tf.GradientTape() as tape:
        _, _, predicted_output = forward_pass_binary(X)
        loss = binary_cross_entropy(y, predicted_output)

    gradients = tape.gradient(loss, [weights_input_hidden, weights_hidden_output_binary])

    weights_input_hidden.assign_sub(learning_rate * gradients[0])
    weights_hidden_output_binary.assign_sub(learning_rate * gradients[1])

X_binary = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 1, 0],
                     [1, 1, 1]], dtype=np.float32)

y_binary = np.array([[0],
                     [0],
                     [0],
                     [1],
                     [0],
                     [1],
                     [1]], dtype=np.float32)


hidden_layer_input, hidden_layer_output, predicted_outputs_binary = forward_pass_binary(X_binary)
loss_binary = binary_cross_entropy(y_binary, predicted_outputs_binary)
print("Перед обновлением весов:")
print("Предсказанные значения:", predicted_outputs_binary.numpy())
print(f'Потери до обновления: {loss_binary.numpy():.4f}')

# Обратное распространение
backward_pass(X_binary, y_binary)

# Повторное
hidden_layer_input, hidden_layer_output, predicted_outputs_binary = forward_pass_binary(X_binary)
new_loss_binary = binary_cross_entropy(y_binary, predicted_outputs_binary)
print("\nПосле обновления весов:")
print("Предсказанные значения:", predicted_outputs_binary.numpy())
print(f'Потери после обновления: {new_loss_binary.numpy():.4f}')
