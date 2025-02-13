import argparse
import numpy as np
from math import exp
from numpy import genfromtxt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--eta")
    parser.add_argument("--iterations")
    args = parser.parse_args()
    fileName = args.data
    learning_rate = float(args.eta)
    iterations = int(args.iterations)
    return fileName, learning_rate, iterations


def prepare_data():
    sample = genfromtxt(file, delimiter=',', autostrip=True).astype(float)
    sample = np.round(sample, 5)
    target = sample[:, -1].astype(float)
    training_examples = sample[:, :-1].astype(float)
    return training_examples, target


def init_h1_weights():
    w_bias_h1, w_a_h1, w_b_h1 = 0.2, -0.3, 0.4
    print(round(w_bias_h1, 5), end=" ")
    print(round(w_a_h1, 5), end=" ")
    print(round(w_b_h1, 5), end=" ")
    return w_bias_h1, w_a_h1, w_b_h1


def init_h2_weights():
    w_bias_h2, w_a_h2, w_b_h2 = -0.5, -0.1, -0.4
    print(round(w_bias_h2, 5), end=" ")
    print(round(w_a_h2, 5), end=" ")
    print(round(w_b_h2, 5), end=" ")
    return w_bias_h2, w_a_h2, w_b_h2


def init_h3_weights():
    w_bias_h3, w_a_h3, w_b_h3 = 0.3, 0.2, 0.1
    print(round(w_bias_h3, 5), end=" ")
    print(round(w_a_h3, 5), end=" ")
    print(round(w_b_h3, 5), end=" ")
    return w_bias_h3, w_a_h3, w_b_h3


def init_ouput_weights():
    w_bias_o, w_h1_o, w_h2_o, w_h3_o = -0.1, 0.1, 0.3, -0.4
    print(round(w_bias_o, 5), end=" ")
    print(round(w_h1_o, 5), end=" ")
    print(round(w_h2_o, 5), end=" ")
    print(round(w_h3_o, 5))
    return w_bias_o, w_h1_o, w_h2_o, w_h3_o


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def main(training_examples, target, iterations, learning_rate):
    init_display()
    w_bias_h1, w_a_h1, w_b_h1 = init_h1_weights()
    w_bias_h2, w_a_h2, w_b_h2 = init_h2_weights()
    w_bias_h3, w_a_h3, w_b_h3 = init_h3_weights()
    w_bias_o, w_h1_o, w_h2_o, w_h3_o = init_ouput_weights()

    for iteration in range(iterations):
        for i in range(len(training_examples)):

            print(training_examples[i][0], end=" ")
            print(training_examples[i][1], end=" ")

            h1_activation = w_a_h1 * training_examples[i][0] + \
                w_b_h1 * training_examples[i][1] + w_bias_h1
            h1_output = sigmoid(h1_activation)

            h2_activation = w_a_h2 * training_examples[i][0] + \
                w_b_h2 * training_examples[i][1] + w_bias_h2
            h2_output = sigmoid(h2_activation)

            h3_activation = w_a_h3 * training_examples[i][0] + \
                w_b_h3 * training_examples[i][1] + w_bias_h3
            h3_output = sigmoid(h3_activation)

            O_activation = h1_output * w_h1_o + h2_output * \
                w_h2_o + h3_output * w_h3_o + w_bias_o
            O_output = sigmoid(O_activation)

            print(round(h1_output, 5), end=" ")
            print(round(h2_output, 5), end=" ")
            print(round(h3_output, 5), end=" ")
            print(round(O_output, 5), end=" ")
            print(int(target[i]), end=" ")

            delta_o = O_output * (1-O_output) * (target[i]-O_output)
            delta_h1 = h1_output * (1-h1_output) * (delta_o * w_h1_o)
            delta_h2 = h2_output * (1-h2_output) * (delta_o * w_h2_o)
            delta_h3 = h3_output * (1-h3_output) * (delta_o * w_h3_o)
            print(round(delta_h1, 5), end=" ")
            print(round(delta_h2, 5), end=" ")
            print(round(delta_h3, 5), end=" ")
            print(round(delta_o, 5), end=" ")

            w_bias_h1 = w_bias_h1 + learning_rate * delta_h1
            w_a_h1 = w_a_h1 + learning_rate * \
                delta_h1 * training_examples[i][0]
            w_b_h1 = w_b_h1 + learning_rate * \
                delta_h1 * training_examples[i][1]
            print(round(w_bias_h1, 5), end=" ")
            print(round(w_a_h1, 5), end=" ")
            print(round(w_b_h1, 5), end=" ")

            w_bias_h2 = w_bias_h2 + learning_rate * delta_h2
            w_a_h2 = w_a_h2 + learning_rate * \
                delta_h2 * training_examples[i][0]
            w_b_h2 = w_b_h2 + learning_rate * \
                delta_h2 * training_examples[i][1]
            print(round(w_bias_h2, 5), end=" ")
            print(round(w_a_h2, 5), end=" ")
            print(round(w_b_h2, 5), end=" ")

            w_bias_h3 = w_bias_h3 + learning_rate * delta_h3
            w_a_h3 = w_a_h3 + learning_rate * \
                delta_h3 * training_examples[i][0]
            w_b_h3 = w_b_h3 + learning_rate * \
                delta_h3 * training_examples[i][1]
            print(round(w_bias_h3, 5), end=" ")
            print(round(w_a_h3, 5), end=" ")
            print(round(w_b_h3, 5), end=" ")

            w_bias_o = w_bias_o + learning_rate * delta_o
            w_h1_o = w_h1_o + learning_rate * delta_o * h1_output
            w_h2_o = w_h2_o + learning_rate * delta_o * h2_output
            w_h3_o = w_h3_o + learning_rate * delta_o * h3_output
            print(round(w_bias_o, 5), end=" ")
            print(round(w_h1_o, 5), end=" ")
            print(round(w_h2_o, 5), end=" ")
            print(round(w_h3_o, 5))


def init_display():
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    global file, learning_rate, iterations
    file, learning_rate, iterations = parse_args()
    training_examples, target = prepare_data()
    main(training_examples, target, iterations, learning_rate)
