import sklearn
import numpy as np
import random

set_order = {'training': 0, 'validating': 1, 'testing': 2, 'all': 3}
word_pos_neg = {'real': 1, 'fake': 0}


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
                all_words[word] = [0.0, 0.0]
    return all_words


def words_count(sets, expect):
    all_words = get_words(sets)
    for i in range(len(sets)):
        for word in sets[i].strip().split():
            if expect[i] == 1:
                all_words[word][1] += 1.0
            else:
                all_words[word][0] += 1.0
    return all_words


def generate_sets():
    random.seed(0)
    all_headlines, real_lines, fake_lines = get_lines()
    training = []
    validating = []
    testing = []
    allset = []
    expected = [[], [], [], []]
    random.shuffle(all_headlines)
    index = 0
    for line in all_headlines:
        if index < len(all_headlines) * 0.7:
            if line in real_lines:
                expected[0].append(1)
            if line in fake_lines:
                expected[0].append(0)
            training.append(line)

        if len(all_headlines) * 0.7 < index < len(all_headlines) * 0.15 + len(all_headlines) * 0.7:
            if line in real_lines:
                expected[1].append(1)
            if line in fake_lines:
                expected[1].append(0)
            validating.append(line)

        if len(all_headlines) * 0.15 + len(all_headlines) * 0.7 < index < len(all_headlines):
            if line in real_lines:
                expected[2].append(1)
            if line in fake_lines:
                expected[2].append(0)
            testing.append(line)

        if line in real_lines:
            expected[3].append(1)
        if line in fake_lines:
            expected[3].append(0)

        allset.append(line)

        index += 1

    return training, validating, testing, allset, expected


def top_words(sets, expect):
    all_words = words_count(sets, expect)
    top = {}
    for word, c in all_words.items():
        top[word] = c[1] + c[0]
    words = sorted(top, key=top.get, reverse=True)[:3]
    return words, all_words


def log_prob_words_given_C(line, words_count, pos_count, neg_count, m, p, cls):
    sumlog = 0
    word_list = line.strip().split()
    for word in word_list:
        if cls == 1:
            prob_word = np.log((words_count[word][cls] + m * p) / (pos_count + p))
        else:
            prob_word = np.log((words_count[word][cls] + m * p) / (neg_count + p))
        sumlog += prob_word
    return sumlog


def log_prob_C_given_words(line, words_count, pos_count, neg_count, m, p, cls):
    log_prob_words_cls = log_prob_words_given_C(line, words_count, pos_count, neg_count, m, p, cls)
    if cls == 1:
        prob_C = pos_count / (pos_count + neg_count)
    else:
        prob_C = neg_count / (neg_count + pos_count)
    return log_prob_words_cls + np.log(prob_C)


def predict(line, words_count, pos_count, neg_count, m, p):
    prob_pos = log_prob_C_given_words(line, words_count, pos_count, neg_count, m, p, 1)
    prob_neg = log_prob_C_given_words(line, words_count, pos_count, neg_count, m, p, 0)
    if prob_pos >= prob_neg:
        return 1
    else:
        return 0


def get_performance(sets, expect, m, p,validating,validating_expect):
    words_c = words_count(validating, validating_expect)
    results = []
    for line in sets:
        results.append(predict(line, words_c, expect.count(1), expect.count(0), m, p))
    return np.sum(np.array(results) == np.array(expect)) / (len(expect) * 1.0)


if __name__ == '__main__':
    training, validating, testing, allset, expected = generate_sets()
    topwords, all_words = top_words(allset, expected[3])
    print(topwords)
    m = 1.0
    p = 0.05
    print(get_performance(validating, expected[1], m, p,validating, expected[1]))
    print(get_performance(testing, expected[2], m, p,validating, expected[1]))
    # print(get_performance(training, expected[0], m, p))
    # print(get_performance(allset, expected[3], m, p))
