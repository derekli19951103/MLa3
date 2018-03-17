from sklearn import tree
from logistic import get_words, generate_single_x
from naivebayes import generate_sets
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from subprocess import call


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
    words_order = [w for w in all_words.keys()]
    i = 0
    for line in sets:
        for word in line.strip().split():
            if word in words_dict:
                words_dict[word] = 1.0
        x[:, i] = generate_single_x(words_dict)
        i += 1
        words_dict = all_words.copy()
    y = np.array(expected[setnum]).reshape((len(expected[setnum]), 1))
    x = x.T
    return x, y,words_order


def train(x, y, maxdepth, split):
    clf = tree.DecisionTreeClassifier(splitter=split, presort=True, max_depth=maxdepth,criterion='entropy')
    clf = clf.fit(x, y)
    return clf


def predict(clf, x, y):
    result = clf.predict(x)
    return np.sum(np.array(result).reshape((len(result), 1)) == y) / len(y)


if __name__ == "__main__":
    train_x, train_y ,words_order= generate_x_y(0)
    val_x, val_y ,_= generate_x_y(1)
    # depth = []
    # for d in range(100, train_x.shape[0], 100):
    #     depth.append(d)
    # depth.append(train_x.shape[0])
    # t_performance = []
    # v_performance = []
    # for d in depth:
    #     clf = train(train_x, train_y, d,'best')
    #     t_performance.append(predict(clf, train_x, train_y))
    #     v_performance.append(predict(clf, val_x, val_y))
    #
    # plt.subplot(1, 2, 1)
    # plt.step(depth, t_performance, label='Training set')
    # plt.step(depth, v_performance, label='Validating set')
    # plt.title('Using best splitter')
    # plt.xlabel('max depth')
    # plt.ylabel("Performance")
    # plt.legend()
    # print('best maxdepth for best splitter:',depth[np.argmax(v_performance)])
    # print('its validating performance:',v_performance[np.argmax(v_performance)])
    #
    # t_performance = []
    # v_performance = []
    # for d in depth:
    #     clf = train(train_x, train_y, d, 'random')
    #     t_performance.append(predict(clf, train_x, train_y))
    #     v_performance.append(predict(clf, val_x, val_y))
    #
    # plt.subplot(1, 2, 2)
    # plt.step(depth, t_performance, label='Training set')
    # plt.step(depth, v_performance, label='Validating set')
    # plt.title('Using random splitter')
    # plt.xlabel('max depth')
    # plt.ylabel("Performance")
    # plt.legend()
    #
    # print('best maxdepth for random splitter:', depth[np.argmax(v_performance)])
    # print('its validating performance:', v_performance[np.argmax(v_performance)])
    #
    # plt.suptitle('Part 7a')
    # plt.savefig("part7a.jpg")
    # plt.gca().clear()
    # print('performances saved')
    clf = train(train_x, train_y, 200, 'best')
    tree.export_graphviz(clf, out_file='tree.dot', feature_names=words_order,max_depth=3)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

