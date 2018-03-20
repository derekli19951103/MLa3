from sklearn import tree
from logistic import get_words, generate_single_x
from naivebayes import generate_sets
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call


def p_word(sets):
    all_words = get_words(sets)
    for i in range(len(sets)):
        for word in sets[i].strip().split():
            all_words[word] += 1.0
    sum_count = sum([v for v in all_words.values()])
    for w, c in all_words.items():
        all_words[w] = (1.0 * all_words[w]) / sum_count
    return all_words


def get_probability(prob_dict):
    for w, p in prob_dict.items():
        prob_dict[w] = np.exp(p)
    return prob_dict


def get_entropy(prob_dict):
    entropy_dict = {}
    for w, p in prob_dict.items():
        entropy_dict[w] = -prob_dict[w] * np.log2(prob_dict[w])
    return entropy_dict


def get_conditional_entropy(cond_prob_dict, p_C):
    cond_prob = get_probability(cond_prob_dict)
    entropy_of_cond = get_entropy(cond_prob)
    cond_entropy_dict = {}
    for w, e in entropy_of_cond.items():
        cond_prob_dict[w] = p_C * entropy_of_cond[w]
    return cond_entropy_dict


def mutual_information(sets, lpwr, lpwf, preal, pfake):
    irw = {}
    ifw = {}
    word_prob_dict = p_word(sets)
    hw = get_entropy(word_prob_dict)
    pfw = get_probability(lpwf)
    prw = get_probability(lpwr)
    hwf = get_conditional_entropy(pfw, pfake)
    hwr = get_conditional_entropy(prw, preal)
    for w, h in hw.items():
        irw[w] = hw[w] - hwr[w]
        ifw[w] = hw[w] - hwf[w]
    return irw, ifw


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
    return x, y, words_order


def train(x, y, maxdepth, split):
    clf = tree.DecisionTreeClassifier(splitter=split, presort=True, max_depth=maxdepth, criterion='entropy')
    clf = clf.fit(x, y)
    return clf


def random_train_top(x, y, maxweights):
    clf = tree.DecisionTreeClassifier(splitter='best', max_depth=1, criterion='entropy', max_features=maxweights)
    clf = clf.fit(x, y)
    return clf


def predict(clf, x, y):
    result = clf.predict(x)
    return np.sum(np.array(result).reshape((len(result), 1)) == y) / len(y)


if __name__ == "__main__":
    print("======================part7======================")
    train_x, train_y, words_order = generate_x_y(0)
    test_x, test_y, _ = generate_x_y(2)
    val_x, val_y, _ = generate_x_y(1)
    depth = []
    for d in range(100, train_x.shape[0], 100):
        depth.append(d)
    depth.append(train_x.shape[0])
    t_performance = []
    v_performance = []
    for d in depth:
        clf = train(train_x, train_y, d, 'best')
        t_performance.append(predict(clf, train_x, train_y))
        v_performance.append(predict(clf, val_x, val_y))

    plt.subplot(1, 2, 1)
    plt.step(depth, t_performance, label='Training set')
    plt.step(depth, v_performance, label='Validating set')
    plt.title('Using best splitter')
    plt.xlabel('max depth')
    plt.ylabel("Performance")
    plt.legend()
    print('best maxdepth for best splitter:', depth[np.argmax(v_performance)])
    print('its validating performance:', v_performance[np.argmax(v_performance)])

    t_performance = []
    v_performance = []
    for d in depth:
        clf = train(train_x, train_y, d, 'random')
        t_performance.append(predict(clf, train_x, train_y))
        v_performance.append(predict(clf, val_x, val_y))

    plt.subplot(1, 2, 2)
    plt.step(depth, t_performance, label='Training set')
    plt.step(depth, v_performance, label='Validating set')
    plt.title('Using random splitter')
    plt.xlabel('max depth')
    plt.ylabel("Performance")
    plt.legend()

    print('best maxdepth for random splitter:', depth[np.argmax(v_performance)])
    print('its validating performance:', v_performance[np.argmax(v_performance)])

    plt.suptitle('Part 7a')
    plt.savefig("part7a.jpg")
    plt.gca().clear()
    print('performances saved')
    clf = train(train_x, train_y, 200, 'best')
    tree.export_graphviz(clf, out_file='tree.dot', feature_names=words_order, max_depth=3, class_names=['fake', 'real'])
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

    print("======================part8======================")
    clf = random_train_top(train_x, train_y, 100)
    tree.export_graphviz(clf, out_file='tree0.dot', feature_names=words_order, max_depth=1,
                         class_names=['fake', 'real'])
    call(['dot', '-Tpng', 'tree0.dot', '-o', 'tree0.png'])
    clf = random_train_top(train_x, train_y, 100)
    tree.export_graphviz(clf, out_file='tree1.dot', feature_names=words_order, max_depth=1,
                         class_names=['fake', 'real'])
    call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])

    print('decision tree summary:')
    print('training accuracy:', predict(clf, train_x, train_y))
    print('validating accuracy:', predict(clf, val_x, val_y))
    print('testing accuracy:', predict(clf, test_x, test_y))
