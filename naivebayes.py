import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

set_order = {'training': 0, 'validating': 1, 'testing': 2, 'all': 3}
word_pos_neg = {'real': 1, 'fake': 0}
top10_real = ['trump', 'donald', 'to', 'us', 'on', 'trumps', 'in', 'of', 'for', 'says']
top10_fake = ['trump', 'the', 'to', 'in', 'donald', 'for', 'of', 'a', 'and', 'on']
top10_real_nw = ['ron', 'neocons', 'watch', 'shameless', 'instantly', 'regrets', 'fleeing', 'dreams', 'secrecy',
                 'faked']
top10_fake_nw = ['glass', 'snl', 'skit', 'korea', 'awkward', 'handshakes', 'g20', 'agenda', 'scouts', 'aides']


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


def get_prob_words_given_C(words_counts, C, num_C, m, p):
    xi_c = {}
    for word, count in words_counts.items():
        xi_c[word] = (min(count[C], num_C) + m * p) / (num_C + m)

    return xi_c


def get_log_sums(small_nums):
    log_sum = 0
    for small_num in small_nums:
        log_sum += np.log(small_num)
    return log_sum


def log_prob_of_hl_given_C(xi_c, line):
    log_prob = np.empty([len(xi_c)])
    i = 0
    for word, prob in xi_c.items():
        if word in line.strip().split():
            log_prob[i] = prob
        else:
            log_prob[i] = 1 - prob
        i += 1

    return log_prob


def get_C(preal, pfake, pwordr, pwordf, sets):
    Cfake = np.array([np.log(pfake) + get_log_sums(log_prob_of_hl_given_C(pwordf, hl)) for hl in sets])
    Creal = np.array([np.log(preal) + get_log_sums(log_prob_of_hl_given_C(pwordr, hl)) for hl in sets])
    C = np.vstack((Cfake, Creal))
    results = np.argmax(C, axis=0)
    return results


def train(sets, expect, m, p, expected, training):
    words_counts = words_count(training, expected[0])
    num_real_data = expected[0].count(1)
    num_fake_data = expected[0].count(0)
    preal = num_real_data / float(num_real_data + num_fake_data)
    pfake = 1 - preal
    pwordr = get_prob_words_given_C(words_counts, 1, num_real_data, m, p)
    pwordf = get_prob_words_given_C(words_counts, 0, num_fake_data, m, p)
    prediction = get_C(preal, pfake, pwordr, pwordf, sets)
    validating_accuracy = np.sum(prediction == np.array(expect)) / float(len(expect))
    return validating_accuracy


def find_m_p(sets, expect, expected, training):
    ms = [1.0, 2.0, 3.0]
    ps = [0.05, 0.1]
    params = []
    accuracy = []
    for m in ms:
        for p in ps:
            validating_accuracy = train(sets, expect, m, p, expected, training)
            params.append((m, p))
            accuracy.append(validating_accuracy)
    return params, accuracy


def get_performance(sets, expect, m, p, expected, training):
    words_counts = words_count(training, expected[0])
    num_real_data = expected[0].count(1)
    num_fake_data = expected[0].count(0)
    preal = num_real_data / float(num_real_data + num_fake_data)
    pfake = 1 - preal
    pwordr = get_prob_words_given_C(words_counts, 1, num_real_data, m, p)
    pwordf = get_prob_words_given_C(words_counts, 0, num_fake_data, m, p)
    prediction = get_C(preal, pfake, pwordr, pwordf, sets)
    accuracy = np.sum(prediction == np.array(expect)) / float(len(expect))
    return accuracy


def get_performance_comparing(sets, expect, m, p, expected, training, p_or_a):
    words_counts = words_count(training, expected[0])
    num_real_data = expected[0].count(1)
    num_fake_data = expected[0].count(0)
    preal = num_real_data / float(num_real_data + num_fake_data)
    pfake = 1 - preal
    pwordr = get_prob_words_given_C(words_counts, 1, num_real_data, m, p)
    pwordf = get_prob_words_given_C(words_counts, 0, num_fake_data, m, p)

    # test part
    if p_or_a == 'p':

        for i in range(len(top10_real)):
            pwordr[top10_real[i]] = pwordr[top10_real_nw[i]]

        for i in range(len(top10_fake)):
            pwordf[top10_fake[i]] = pwordf[top10_fake_nw[i]]

    if p_or_a == 'a':

        for i in range(len(top10_real)):
            pwordr[top10_real_nw[i]] = pwordr[top10_real[i]]

        for i in range(len(top10_fake)):
            pwordf[top10_fake_nw[i]] = pwordf[top10_fake[i]]

    # test part
    prediction = get_C(preal, pfake, pwordr, pwordf, sets)
    accuracy = np.sum(prediction == np.array(expect)) / float(len(expect))
    return accuracy


def get_C_word(preal, pfake, pwordr, pwordf, sets):
    Cfake = np.array([np.log(pfake) + get_log_sums(log_prob_of_hl_given_C(pwordf, hl)) for hl in sets])
    Creal = np.array([np.log(preal) + get_log_sums(log_prob_of_hl_given_C(pwordr, hl)) for hl in sets])
    return Cfake, Creal


def get_C_nonword(preal, pfake, pwordr, pwordf, sets):
    Cfake = np.array([np.log(pfake) + pwordf[hl] for hl in sets])
    Creal = np.array([np.log(preal) + pwordr[hl] for hl in sets])
    return Cfake, Creal


def part3a(p_fake_w, p_real_w, p_fake_nw, p_real_nw, sets, m, p, expected, training):
    '''
    :param sets: list of words
    :param expect:
    :param m:
    :param p:
    :param expected:
    :param training:
    :return: dictionary P(real | word) and P(fake | word) for words.
    '''

    words_counts = words_count(training, expected[0])
    num_real_data = expected[0].count(1)
    num_fake_data = expected[0].count(0)

    # P(real)
    preal = num_real_data / float(num_real_data + num_fake_data)
    # P(fake)
    pfake = 1 - preal
    # P(word | real)
    pwordr = get_prob_words_given_C(words_counts, 1, num_real_data, m, p)
    # P(word | fake)
    pwordf = get_prob_words_given_C(words_counts, 0, num_fake_data, m, p)

    # P(not word | real)
    pnwordr = pwordr.copy()
    sum_pnwordr = sum([item for key, item in pwordr.items()])
    for key, item in pnwordr.items():
        pnwordr[key] = sum_pnwordr - pnwordr[key]

    # P(not word | fake)
    pnwordf = pwordf.copy()
    sum_pnwordf = sum([item for key, item in pwordf.items()])
    for key, item in pnwordf.items():
        pnwordf[key] = sum_pnwordf - pnwordf[key]

    for the_word in sets:
        Cfake, Creal = get_C_word(preal, pfake, pwordr, pwordf, [the_word])
        # P(fake | word)
        p_fake_w[the_word] = Cfake[0]
        # P(real | word)
        p_real_w[the_word] = Creal[0]

        Cfaken, Crealn = get_C_nonword(preal, pfake, pnwordr, pnwordf, [the_word])
        # P(real | not word)
        p_fake_nw[the_word] = Cfaken[0]
        # P(fake | not word)
        p_real_nw[the_word] = Crealn[0]

    return p_fake_w, p_real_w, p_fake_nw, p_real_nw


if __name__ == '__main__':
    training, validating, testing, allset, expected = generate_sets()
    topwords, all_words = top_words(allset, expected[3])
    p_real_w = {}
    p_fake_w = {}
    p_real_nw = {}
    p_fake_nw = {}
    print("===================part1===================")
    i = 1
    for w in topwords:
        print('Top{} word: '.format(i), w)
        print('appearance in fake:', all_words[w][0])
        print('appearance in real:', all_words[w][1])
        i += 1
    print("===================part2===================")
    # find mp
    para, accu = find_m_p(validating, expected[1], expected, training)
    x_axis = [i + 1 for i in range(len(accu))]
    plt.plot(x_axis, accu)
    plt.title('Part 2')
    plt.xlabel('M and P')
    plt.ylabel("Performance")
    plt.savefig("part2.jpg")
    plt.close("all")

    # print out mp set and its performance
    for i in range(len(x_axis)):
        print("****************************")
        print("# of set: ", x_axis[i])
        print("M and P: ", para[i])
        print("Performance: ", accu[i])
    print("****************************")

    m = 2.0
    p = 0.1

    print('training accuracy:', get_performance(training, expected[0], m, p, expected, training))
    print('validating accuracy:', get_performance(validating, expected[1], m, p, expected, training))
    print('testing accuracy:', get_performance(testing, expected[2], m, p, expected, training))

    print("===================part3===================")
    p_real_w = {}
    p_fake_w = {}
    p_real_nw = {}
    p_fake_nw = {}
    # P(real) and P(fake)

    word_set = [word for word in get_words(training).keys()]
    p_fake_w, p_real_w, p_fake_nw, p_real_nw = part3a(p_fake_w, p_real_w, p_fake_nw, p_real_nw, word_set, m, p,
                                                      expected, training)

    print("P(real | word)")
    print(sorted(p_real_w, key=p_real_w.get, reverse=True)[:10])
    print("P(real | not word).")
    print(sorted(p_real_nw, key=p_real_nw.get, reverse=True)[:10])
    print("P(fake | word)")
    print(sorted(p_fake_w, key=p_fake_w.get, reverse=True)[:10])
    print("P(fake | not word)")
    print(sorted(p_fake_nw, key=p_fake_nw.get, reverse=True)[:10])

    # print("****************************")

    print('presence->absence:')
    print('training accuracy:', get_performance_comparing(training, expected[0], m, p, expected, training, 'p'))
    print('validating accuracy:', get_performance_comparing(validating, expected[1], m, p, expected, training, 'p'))
    print('testing accuracy:', get_performance_comparing(testing, expected[2], m, p, expected, training, 'p'))
    print('absence->presence:')
    print('training accuracy:', get_performance_comparing(training, expected[0], m, p, expected, training, 'a'))
    print('validating accuracy:', get_performance_comparing(validating, expected[1], m, p, expected, training, 'a'))
    print('testing accuracy:', get_performance_comparing(testing, expected[2], m, p, expected, training, 'a'))

    nonstop_p_real_w = p_real_w.copy()
    nonstop_p_fake_w = p_fake_w.copy()

    for word in ENGLISH_STOP_WORDS:
        if word in nonstop_p_real_w.keys():
            nonstop_p_real_w.pop(word)
        if word in nonstop_p_fake_w.keys():
            nonstop_p_fake_w.pop(word)
    print("****************************")
    print("P(real | word) in Non-stop word")
    print(sorted(nonstop_p_real_w, key=nonstop_p_real_w.get, reverse=True)[:10])
    print("P(fake | word) in Non-stop word")
    print(sorted(nonstop_p_fake_w, key=nonstop_p_fake_w.get, reverse=True)[:10])

    print('naive bayes summary:')
    tr = get_performance(training, expected[0], m, p, expected, training)
    v = get_performance(validating, expected[1], m, p, expected, training)
    te = get_performance(testing, expected[2], m, p, expected, training)
    print('training accuracy:', tr)
    print('validating accuracy:', v)
    print('testing accuracy:', te)
