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
    all_words = get_words(all_headlines)
    training = []
    validating = []
    testing = []
    count_c = {'training': [0, 0], 'validating': [0, 0], 'testing': [0, 0]}
    word_count = {'training': {'real': all_words, 'fake': all_words},
                  'validating': {'real': all_words, 'fake': all_words},
                  'testing': {'real': all_words, 'fake': all_words}}
    random.shuffle(all_headlines)
    index = 0
    for line in all_headlines:
        if index < len(all_headlines) * 0.7:
            if line in real_lines:
                count_c['training'][0] += 1.0
                for word in line.strip().split():
                    word_count['training']['real'][word] += 1.0
            if line in fake_lines:
                count_c['training'][1] += 1.0
                for word in line.strip().split():
                    word_count['training']['fake'][word] += 1.0
            training.append(line)

        if len(all_headlines) * 0.7 < index < len(all_headlines) * 0.15 + len(all_headlines) * 0.7:
            if line in real_lines:
                count_c['validating'][0] += 1.0
                for word in line.strip().split():
                    word_count['validating']['real'][word] += 1.0
            if line in fake_lines:
                count_c['validating'][1] += 1.0
                for word in line.strip().split():
                    word_count['validating']['fake'][word] += 1.0
            validating.append(line)

        if len(all_headlines) * 0.15 + len(all_headlines) * 0.7 < index < len(all_headlines):
            if line in real_lines:
                count_c['testing'][0] += 1.0
                for word in line.strip().split():
                    word_count['testing']['real'][word] += 1.0
            if line in fake_lines:
                count_c['testing'][1] += 1.0
                for word in line.strip().split():
                    word_count['testing']['fake'][word] += 1.0
            testing.append(line)

        index += 1

    return training, validating, testing, count_c, word_count


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


def train(setname):
    training, validating, testing, count_c, word_count = generate_sets()
    MAP = []
    pi = 0
    mean = np.mean([value for value in word_count[setname]['real'].values()])
    for word, count_xi in word_count[setname]['real'].items():
        pi += np.log((count_xi + mean * 2.0) / (count_c[setname][0] + mean))
    pi += np.log(count_c[setname][0])
    MAP.append(np.exp(pi))
    print(pi)
    pi = 0
    mean = np.mean([value for value in word_count[setname]['fake'].values()])
    for word, count_xi in word_count[setname]['fake'].items():
        pi += np.log((count_xi + mean * 2.0) / (count_c[setname][1] + mean))
    pi += np.log(count_c[setname][1])
    MAP.append(np.exp(pi))
    return np.argmax(MAP)


if __name__ == '__main__':
    random.seed(0)
    print(top_words())
    print(train('validating'))
