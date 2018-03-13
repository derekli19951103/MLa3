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


def get_words(all_headlines):
    all_words = {}
    for line in all_headlines:
        for word in line.strip().split():
            if word not in all_words:
                all_words[word] = 0.0
    return all_words


def generate_sets():
    all_headlines, real_lines, fake_lines = get_lines()
    training = []
    validating = []
    testing = []
    count_c = {'training': [0, 0], 'validating': [0, 0], 'testing': [0, 0]}
    expected = {'training': [], 'validating': [], 'testing': []}
    random.shuffle(all_headlines)
    index = 0
    for line in all_headlines:
        if index < len(all_headlines) * 0.7:
            if line in real_lines:
                count_c['training'][0] += 1.0
                expected['training'].append(1)
            if line in fake_lines:
                count_c['training'][1] += 1.0
                expected['training'].append(0)
            training.append(line)

        if len(all_headlines) * 0.7 < index < len(all_headlines) * 0.15 + len(all_headlines) * 0.7:
            if line in real_lines:
                count_c['validating'][0] += 1.0
                expected['validating'].append(1)
            if line in fake_lines:
                count_c['validating'][1] += 1.0
                expected['validating'].append(0)
            validating.append(line)

        if len(all_headlines) * 0.15 + len(all_headlines) * 0.7 < index < len(all_headlines):
            if line in real_lines:
                count_c['testing'][0] += 1.0
                expected['testing'].append(1)
            if line in fake_lines:
                count_c['testing'][1] += 1.0
                expected['testing'].append(0)
            testing.append(line)

        index += 1

    return training, validating, testing, count_c, expected


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
    return [(w, all_words[w]) for w in words]


def train(setname, m, p):
    training, validating, testing, count_c, expected = generate_sets()
    all_words = get_words(validating)
    words = all_words.copy()
    results = []
    for line in validating:
        for word in line.strip().split():
            words[word] = 1.0
        MAP = []
        pi = 0
        for w, t in words.items():
            pi += np.log((t + m * p) / (count_c[setname][0] + m))
        pi += np.log(count_c[setname][0])
        MAP.append(np.exp(pi))
        pi = 0
        for w, t in words.items():
            pi += np.log((t + m * p) / (count_c[setname][1] + m))
        pi += np.log(count_c[setname][1])
        MAP.append(np.exp(pi))
        result = np.argmax(MAP)
        results.append(result)
        words = all_words.copy()
    return np.sum(np.array(results) == np.array(expected[setname]))/(len(expected[setname])*1.0)


if __name__ == '__main__':
    random.seed(0)
    print(top_words())
    for i in np.arange(2.0,3.0,0.05):
        print(train('validating', 10., i))
