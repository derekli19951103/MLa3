from naivebayes import generate_sets
from torch.autograd import Variable
import torch
import numpy as np

label = {'real': [1, 0], 'fake': [0, 1]}
set_order = {'training': 0, 'validating': 1, 'testing': 2, 'all': 3}


def get_words(sets):
    all_words = {}
    for line in sets:
        for word in line.strip().split():
            if word not in all_words:
                all_words[word] = 0.0
    return all_words


def generate_single_x(words_dict):
    x = np.zeros((len(words_dict),))
    i = 0
    for w, v in words_dict.items():
        x[i] = v
        i += 1
    return x


def generate_x_y(setnum):
    training, validating, testing, allset, expected = generate_sets()
    sets = []
    if setnum == 0:
        sets = training
    if setnum == 1:
        sets = validating
    if setnum == 2:
        sets = testing
    if setnum == 3:
        sets = allset
    all_words = get_words(training)
    words_dict = all_words.copy()
    x = np.zeros((len(all_words), len(sets)))
    i = 0
    for line in sets:
        for word in line.strip().split():
            if word in words_dict:
                words_dict[word] = 1.0
        x[:, i] = generate_single_x(words_dict)
        i += 1
        words_dict = all_words.copy()
    y = np.array(expected[setnum])
    temp_y = 1 - y
    y = np.vstack((y, temp_y))
    return x, y


def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model


def train_l2(model, x, y, n):
    torch.manual_seed(1)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    loss_fn = torch.nn.CrossEntropyLoss()

    ## TRAINING THE MODEL
    alpha = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)
    batch_size = 100

    for t in range(n):
        num_batches = x.shape[1] // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            mini_x = Variable(torch.from_numpy(x.T[start:end]), requires_grad=False).type(dtype_float)
            mini_y_labels = Variable(torch.from_numpy(np.argmax(y[:, start:end], 0)), requires_grad=False).type(
                dtype_long)
            # forward
            y_pred = model(mini_x)
            loss = loss_fn(y_pred, mini_y_labels)
            # add l1 regularization
            reg_lambda = 1.0 / x.shape[1]
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg += W.norm(2)
            loss += reg_lambda * l2_reg
            model.zero_grad()
            # print("[Current Loss] ", loss)
            # backward
            loss.backward()
            optimizer.step()

    return model


def train_l1(model, x, y, n):
    torch.manual_seed(1)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    loss_fn = torch.nn.CrossEntropyLoss()

    ## TRAINING THE MODEL
    alpha = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)
    batch_size = 100

    for t in range(n):
        num_batches = x.shape[1] // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            mini_x = Variable(torch.from_numpy(x.T[start:end]), requires_grad=False).type(dtype_float)
            mini_y_labels = Variable(torch.from_numpy(np.argmax(y[:, start:end], 0)), requires_grad=False).type(
                dtype_long)
            # forward
            y_pred = model(mini_x)
            loss = loss_fn(y_pred, mini_y_labels)
            # add l1 regularization
            reg_lambda = 1.0 / x.shape[1]
            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = W.norm(1)
                else:
                    l1_reg += W.norm(1)
            loss += reg_lambda * l1_reg
            model.zero_grad()
            # print("[Current Loss] ", loss)
            # backward
            loss.backward()
            optimizer.step()

    return model


def predict(model, x, y):
    dtype_float = torch.FloatTensor
    x = Variable(torch.from_numpy(x.T), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    performance = np.mean(np.argmax(y_pred, 1) == np.argmax(y, 0))
    return performance


if __name__ == '__main__':
    x, y = generate_x_y(0)
    model_l2 = build_model(x.shape[0], y.shape[0])
    model_l1 = build_model(x.shape[0], y.shape[0])
    model_l2 = train_l2(model_l2, x, y, 200)
    model_l1 = train_l1(model_l1, x, y, 200)
    x, y = generate_x_y(2)
    predY_l2 = predict(model_l2, x, y)
    predY_l1 = predict(model_l1, x, y)
    print('============Trained using training set, test using testing set============')
    print('L2 Regularization accuracy:',predY_l2)
    print('L2 Regularization accuracy:',predY_l1)
