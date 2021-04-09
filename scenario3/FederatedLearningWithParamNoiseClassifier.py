# -*- coding: utf-8 -*-

import struct
import matplotlib.pyplot as plt
import numpy as np
import random
import time


def readMNISTdata():
    global N_client

    with open('../datasets/mnist/t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))

    with open('../datasets/mnist/t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('../datasets/mnist/train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows * ncols))

    with open('../datasets/mnist/train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate((np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate((np.ones([test_data.shape[0], 1]), test_data), axis=1)

    all_data = np.concatenate((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))

    del train_data
    del train_labels
    del test_data
    del test_labels

    rnd = int(random.random() * 1000)

    np.random.seed(rnd)
    np.random.shuffle(all_labels)
    np.random.seed(rnd)
    np.random.shuffle(all_data)

    # for server: 14K for train, 2K for validation and 4K for test
    server = {'train': (all_data[:14000] / 256, all_labels[:14000]),
              'val': (all_data[14000:16000] / 256, all_labels[14000:16000]),
              'test': (all_data[16000:20000] / 256, all_labels[16000:20000])}

    clients = {}
    for i in range(N_client):
        # for each client: 4K for training and 1K for validation
        clients[i] = {'train': (all_data[20000 + i * 5000:20000 + (i + 1) * 5000 - 1000] / 256,
                                all_labels[20000 + i * 5000:20000 + (i + 1) * 5000 - 1000]),
                      'val': (all_data[20000 + i * 5000 + 4000:20000 + (i + 1) * 5000] / 256,
                              all_labels[20000 + i * 5000 + 4000:20000 + (i + 1) * 5000])}

    return server, clients


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


def train(X_train, y_train, X_val, t_val, W):
    N_train = X_train.shape[0]

    W_best = W
    acc_best = 0
    loss_best = 0

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

        # 2. Perform validation on the validation test by the risk
        y_val, t_hat_val, loss_val, acc_val = predict(X_val, W, t_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if acc_val > acc_best:
            W_best = W
            acc_best = acc_val
            loss_best = avg_loss

    # Return some variables as needed
    return W_best, acc_best, loss_best


def train_server():
    global server
    global clients

    losses_train = []
    accs_train = []
    Ws_clients = []

    Features = server['train'][0].shape[1]

    # initialization
    W_server = np.array([[0] * N_class for _ in range(Features)])

    for l in range(MaxFLIter):
        print("Iteration #", l+1, "of the FL system")
        # train the server
        W_best, acc_best, loss_best = \
            train(server['train'][0], server['train'][1], server['val'][0], server['val'][1], W_server)

        accs_train.append(acc_best)
        losses_train.append(loss_best)

        # train clients based on W from server
        for c in range(N_client):
            W_client, _, _ = \
                train(clients[c]['train'][0], clients[c]['train'][1],
                      clients[c]['val'][0], clients[c]['val'][1], W_best)
            W_client /= max(1.0, np.linalg.norm(W_client, 1)/C)
            W_client += np.random.normal(client_mu, client_sigma*client_sigma, size=W_client.shape)
            Ws_clients.append(W_client)

        # take average of W_client and update W_server
        W_server = np.sum(Ws_clients, axis=0)/N_client
        W_server += np.random.normal(server_mu, server_sigma*server_sigma, size=W_server.shape)

    return W_server, accs_train, losses_train


##############################
N_class = 10
N_client = 10
C = 20

client_mu = 0
client_sigma = 0.031075

server_mu = 0
server_sigma = 0.6137

alpha = 0.1  # learning rate
batch_size = 100  # batch size
MaxIter = 30  # Maximum iteration
MaxFLIter = 20

# Main code starts here
server, clients = readMNISTdata()

start = time.time()
W_server, accs, losses = train_server()
end = time.time()

print('time takes to train (s)', (end - start))
# time takes to train (s) 492.78326392173767

_, _, loss, acc = predict(server['test'][0], W_server, server['test'][1])

print('For the test data: loss=', loss, 'and acc=', acc)
# For the test data: loss= 0.07112466239947182 and acc= 0.81375

plt.figure()
plt.plot([x for x in range(MaxFLIter)], accs, color="blue", label="Accuracy")
plt.plot([x for x in range(MaxFLIter)], losses, color="red", label="Training Loss")
plt.xlabel('number of epochs')
plt.legend()
plt.tight_layout()
plt.savefig('federated_learning_param_noise.jpg')

plt.figure()
plt.plot([x for x in range(MaxFLIter)], accs, color="blue", label="Accuracy")
plt.xlabel('number of epochs')
plt.legend()
plt.tight_layout()
plt.savefig('federated_learning_param_noise_accuracy.jpg')

plt.figure()
plt.plot([x for x in range(MaxFLIter)], losses, color="red", label="Training Loss")
plt.xlabel('number of epochs')
plt.legend()
plt.tight_layout()
plt.savefig('federated_learning_param_noise_loss.jpg')
