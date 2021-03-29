# -*- coding: utf-8 -*-


import struct

import matplotlib.pyplot as plt
import numpy as np


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows * ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate((np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate((np.ones([test_data.shape[0], 1]), test_data), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels


def predict(X, W, t=None):
    M = X.shape[0]

    Z = np.dot(X, W)
    Z = Z - np.max(Z, axis=1)[:, None]

    y_hat = np.exp(Z)
    y = y_hat / np.sum(y_hat, axis=1)[:, None]

    t_hat = np.array(y.argmax(axis=1)).T
    y_max = y.max(axis=1)

    acc = (t.flatten() == t_hat).sum() / len(t_hat)

    loss = (-1 / M) * (np.sum(np.log(y_max)))

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    Features = X_train.shape[1]

    # initialization
    W = np.array([[0] * N_class for _ in range(Features)])
    losses_train = []
    accs_val = []

    W_best = W
    acc_best = 0
    acc_train = 0
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_epoch = 0
        epoch_acc = 0
        for b in range(int(np.ceil(N_train / batch_size))):
            X_batch = X_train[b * batch_size: (b + 1) * batch_size]
            y_batch = y_train[b * batch_size: (b + 1) * batch_size]
            t = np.array([[0] * N_class for _ in range(X_batch.shape[0])])
            index = 0
            for yi in y_batch.flatten():
                t[index][yi] = 1
                index += 1

            X_batch_size = X_batch.shape[0]
            y, t_hat, loss, acc = predict(X_batch, W, y_batch)
            loss_epoch += loss
            epoch_acc += acc * X_batch_size

            # Mini-batch gradient descent
            dW = (-1 / X_batch_size) * (np.dot(X_batch.T, t - y))
            W = np.subtract(W, alpha * dW)

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_loss = loss_epoch / int(np.ceil(N_train / batch_size))
        losses_train.append(avg_loss)

        # 2. Perform validation on the validation test by the risk
        y_val, t_hat_val, loss_val, acc_val = predict(X_val, W, t_val)
        accs_val.append(acc_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if acc_val > acc_best:
            W_best = W
            acc_best = acc_val
            epoch_best = epoch
            acc_train = epoch_acc / N_train

    # Return some variables as needed

    return epoch_best, acc_best, acc_train, W_best, accs_val, losses_train


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 10

alpha = 0.1  # learning rate
batch_size = 100  # batch size
MaxIter = 50  # Maximum iteration
decay = 0.  # weight decay

epoch_best, acc_best, acc_train, W_best, accs, losses = train(X_train, t_train, X_val, t_val)

_, _, loss, acc = predict(X_test, W_best, t_test)

print('For the test data: loss=', loss, 'and acc=', acc)
#For the test data: loss= 0.00031469297870585317  and acc= 0.9232

plt.figure()
plt.plot([x for x in range(MaxIter)], accs, color="blue", label="Accuracy")
plt.xlabel('number of epochs')
plt.legend()
plt.tight_layout()
plt.savefig('centralized_server_accuracy.jpg')

plt.figure()
plt.plot([x for x in range(MaxIter)], losses, color="red", label="Training Loss")
plt.xlabel('number of epochs')
plt.legend()
plt.tight_layout()
plt.savefig('centralized_server_loss.jpg')
