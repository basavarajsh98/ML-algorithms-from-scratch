import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--eta")
    parser.add_argument("--threshold")
    args = parser.parse_args()
    fileName = args.data
    learning_rate = float(args.eta)
    threshold = float(args.threshold)
    return fileName, learning_rate, threshold


def prepare_data():
    input_file = pd.read_csv(fileName, header=None)
    training_examples = np.array(input_file, 'float')
    # add bias
    x0 = np.ones((len(training_examples), 1))
    x = np.append(x0, training_examples[:, :2], axis=1)
    y = training_examples[:, -1].reshape(len(x), 1)
    return x, y


def find_SSE(w):
    SSE = np.sum((y - np.dot(x, w)) ** 2)
    return SSE


def find_gradient(w):
    return np.sum(np.multiply(x, np.subtract(y, np.dot(x, w))), axis=0)


def update_weights(w, gradient):
    return np.add(w, (learning_rate * gradient).reshape(w.shape))


def display(iterations, w, SSE):
    print(str(iterations) + "," + format(w[0, 0], '.9f') + "," + format((w[1, 0]), '.9f') + "," + format(w[2, 0], '.9f') + "," + format(SSE, '.9f'))


def linearRegression():
    # initialize weights
    w_old = np.zeros([3, 1])
    # initial error
    SSE_old = find_SSE(w_old)
    iterations = 0
    SSE_diff = 1
    display(iterations, w_old, SSE_old)
    # learn until the threshold is reached
    while SSE_diff >= threshold:
        gradient = find_gradient(w_old)
        w_new = update_weights(w_old, gradient)
        SSE_new = find_SSE(w_new)
        SSE_diff = abs(SSE_new - SSE_old)
        # reinitialize weights and error for the next iteration
        w_old = w_new
        SSE_old = SSE_new
        iterations += 1
        display(iterations, w_new, SSE_new)


if __name__ == "__main__":
    global x, y, fileName, learning_rate, threshold
    fileName, learning_rate, threshold = parse_args()
    x, y = prepare_data()
    linearRegression()
