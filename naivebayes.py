import sklearn
import numpy as np
import random


def get_lines():
    real = open('clean_real.txt', 'r')
    real_lines = real.readlines()
    fake = open('clean_fake.txt', 'r')
    fake_lines = fake.readlines()
    all_headlines = []
    for real_line in real_lines:
        all_headlines.append(real_line)
    for fake_line in fake_lines:
        all_headlines.append(fake_line)
    real.close()
    fake.close()
    return all_headlines, real_lines, fake_lines


def get_words(sets):
    all_words = {}
    for line in sets:
        for word in line.strip().split():
            if word not in all_words:
                all_words[word] = {'real': 0.0, 'fake': 0.0}
    return all_words


def generate_sets():
    all_headlines, real_lines, fake_lines = get_lines()
    training = []
    validating = []
    testing = []
    expected = {'training': [], 'validating': [], 'testing': [], 'all': []}
    random.shuffle(all_headlines)
    index = 0
    for line in all_headlines:
        if index < len(all_headlines) * 0.7:
            if line in real_lines:
                expected['training'].append(1)
            if line in fake_lines:
                expected['training'].append(0)
            training.append(line)

        if len(all_headlines) * 0.7 < index < len(all_headlines) * 0.15 + len(all_headlines) * 0.7:
            if line in real_lines:
                expected['validating'].append(1)
            if line in fake_lines:
                expected['validating'].append(0)
            validating.append(line)

        if len(all_headlines) * 0.15 + len(all_headlines) * 0.7 < index < len(all_headlines):
            if line in real_lines:
                expected['testing'].append(1)
            if line in fake_lines:
                expected['testing'].append(0)
            testing.append(line)

        if line in real_lines:
            expected['all'].append(1)
        if line in fake_lines:
            expected['all'].append(0)
        testing.append(line)

        index += 1

    return training, validating, testing, expected


def top_words():
    all_words = {}
    all_headlines, real_lines, fake_lines = get_lines()
    for line in all_headlines:
        for word in line.strip().split():
            if word not in all_words:
                all_words[word] = 1
            else:
                all_words[word] += 1
    words = sorted(all_words, key=all_words.get, reverse=True)[:3]
    return words


def words_count(setname=None):
    training, validating, testing, expected = generate_sets()
    sets = []
    if setname == None:
        setname='all'
        all_headlines, real_lines, fake_lines = get_lines()
        sets = all_headlines
    if setname == 'training':
        sets = training
    if setname == 'validating':
        sets = validating
    if setname == 'testing':
        sets = testing
    all_words = get_words(sets)
    for i in range(len(sets)):
        for word in sets[i].strip().split():
            if expected[setname][i] == 1:
                all_words[word]['real'] += 1.0
            else:
                all_words[word]['fake'] += 1.0
    return all_words


def train(setname):
    training, validating, testing, expected = generate_sets()
    sets = []
    if setname == 'training':
        sets = training
    if setname == 'validating':
        sets = validating
    if setname == 'testing':
        sets = testing
    all_words = {}
    for line in sets:
        for word in line.strip().split():
            if word not in all_words:
                all_words[word] = 0.0
    words = all_words.copy()
    results = []
    for line in validating:
        for word in line.strip().split():
            words[word] = 1.0
        MAP = []
        pi = np.log(expected[setname].count(1))
        for w, t in words.items():
            pi += np.log((t + 1.0) / (expected[setname].count(1) + 1.0))
        MAP.append(pi)
        pi = np.log(expected[setname].count(0))
        for w, t in words.items():
            pi += np.log((t + 1.0) / (expected[setname].count(0) + 1.0))
        MAP.append(pi)
        result = np.argmax(MAP)
        results.append(result)
        words = all_words.copy()
    return np.sum(np.array(results) == np.array(expected[setname])) / (len(expected[setname]) * 1.0)


if __name__ == '__main__':
    random.seed(0)
    all_words = words_count()
    words = top_words()
    for w in words:
        print(w,all_words[w])
    print(train('validating'))
