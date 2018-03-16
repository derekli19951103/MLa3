from sklearn import tree
from logistic import get_words, generate_single_x
from naivebayes import generate_sets
import numpy as np


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
    y = np.array(expected[setnum]).reshape((len(expected[setnum]), 1))
    x = x.T
    return x, y


if __name__ == "__main__":
    x, y = generate_x_y(0)
    print(x.shape)
    print(y.shape)
    clf = tree.DecisionTreeClassifier(splitter='best',presort=True)
    clf = clf.fit(x, y)
    test,test_y=generate_x_y(1)
    result=clf.predict(test)
    print(len(result))
    print(len(test_y))
    print(np.sum(np.array(result).reshape((len(result),1))==test_y)/len(test_y))


