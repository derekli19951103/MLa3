import numpy as np
import random
import matplotlib.pyplot as plt

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
    ms = [10.0, 1.0, 0.1]
    ps = [10.0, 1.0, 0.1]
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


if __name__ == '__main__':
    training, validating, testing, allset, expected = generate_sets()
    topwords, all_words = top_words(allset, expected[3])
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
    x_axis = [i+1 for i in range(len(accu))]
    plt.plot(x_axis, accu)
    plt.title('Part 2')
    plt.xlabel('M and P')
    plt.ylabel("Performance")
    plt.savefig("part2.jpg")
    plt.close("all")

    # print out mp set and its performance
    for i in range(len(x_axis)):
        print ("****************************")
        print ("# of set: ", x_axis[i])
        print ("M and P: ", para[i])
        print ("Performance: ", accu[i])

    m = 1.0
    p = 0.1
    print('training accuracy:', get_performance(training, expected[0], m, p, expected, training))
    print('validating accuracy:', get_performance(validating, expected[1], m, p, expected, training))
    print('testing accuracy:', get_performance(testing, expected[2], m, p, expected, training))


    print ("===================part3===================")

    # P(real) and P(fake)
    p_real = None
    p_fake = None
    num_real = 0
    num_fake = 0
    for key in all_words.keys():
        num_real += all_words[key][1]
        num_fake += all_words[key][0]

    num_total = num_real + num_fake
    p_real = num_real / num_total
    p_fake = 1 - p_real


    # 10 presence most predict real
    # P(real | word) = P(word | real) P(real) / P(word)


    # 10 absence most predict real
    # P(real | not word) = P(not word | real) P(real) / P(word)

    # 10 presence most predict fake
    # P(fake | word) = P(word | fake) P(fake) / P(word)

    # 10 absence most predict fake
    # P(fake | not word) = P(not word | fake) P(fake) / P(word)
