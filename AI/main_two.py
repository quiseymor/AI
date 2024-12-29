import math
import numpy as np


def wxb(w, x, b):
    return w @ x + b


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x: np.array):
    return [np.exp(i)/sum(np.exp(x)) for i in x]


def relu(x):
    return np.maximum(x, 0.)


def error_func1(yt, yv):
    return -np.mean(yt * np.log(yv) + (1 - yt) * np.log(1 - yv))


def error_func2(n, yt, yv):
    return np.mean(np.square(yv - yt))


class BinaryModel:
    def __init__(self):
        self.input_val = np.array                   # входные
        self.output_val = np.array                  # выходные
        self.eps = 0.001                            # для ошибки
        # данные для модели с одним скрытым слоем
        self.count_inp = 3
        self.count_out = 1
        self.count_ney_hide_layer = 4
        self.W1 = np.random.random((self.count_ney_hide_layer, self.count_inp))     # весовые коэф 1 слой
        self.W2 = np.random.random((self.count_out, self.count_ney_hide_layer))     # весовые коэф 2 слой
        self.B = np.random.random((2, 1))           # коэф смещения

    def set(self, inp: np.array, out: np.array, n_out: int):
        self.input_val = inp
        self.output_val = out
        # если количество классов меняется
        if self.count_out != n_out:
            self.W2 = np.random.random((n_out, self.count_ney_hide_layer))
        self.count_out = n_out

    def show_val(self):
        print(f"input = {self.input_val}\ntarget out = {self.output_val}\n"
              f"W вход-скрыт = {self.W1}\nW скрыт-выход = {self.W2}\nB = {self.B}\n"
              f"Количество классов = {self.count_out}")

    def fit(self):
        out = []
        for i in range(self.count_ney_hide_layer):
            out.append(relu(np.dot(self.input_val, self.W1[i]) + self.B[0, 0]))

        temp = out.copy()
        out.clear()
        if self.count_out == 1:
            for j in range(self.count_out):
                out.append(sigmoid(np.dot(temp, self.W2[j]) + self.B[1, 0]))
            error = error_func2(self.count_out, out, self.output_val)
        else:
            for j in range(self.count_out):
                out.append(np.dot(temp, self.W2[j]) + self.B[1, 0])
            out = np.array(softmax(out))
            error = error_func1(out, self.output_val)
        print(f"result = {out}\nerror = {error}\n")
        return 0


def start():
    x = np.array([[0.05, 0.1, 0.15],
                 [0.2, 0.25, 0.3],
                 [0.35, 0.4, 0.45]])
    y1 = np.array([1.])
    y2 = np.array([0.2, 0.3, 0.5])

    model = BinaryModel()
    for i in range(len(x)):
        model.set(x[i], y1, len(y1))
        model.show_val()
        model.fit()

    for i in range(len(x)):
        model.set(x[i], y2, len(y2))
        model.show_val()
        model.fit()


if __name__ == '__main__':
    start()

