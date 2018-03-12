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


def generate_sets():
    all_headlines, real_lines, fake_lines = get_lines()
    training = []
    validating = []
    testing = []
    random.shuffle(all_headlines)
    index = 0
    for line in all_headlines:
        if index < len(all_headlines) * 0.7:
            training.append(line.strip().split())
        if len(all_headlines) * 0.7 < index < len(all_headlines) * 0.15 + len(all_headlines) * 0.7:
            validating.append(line.strip().split())
        if len(all_headlines) * 0.15 + len(all_headlines) * 0.7 < index < len(all_headlines):
            testing.append(line.strip().split())
        index += 1

    return training, validating, testing


def word_count():
    real_words = {}
    fake_words = {}
    all_headlines, real_lines, fake_lines = get_lines()
    for real_line in real_lines:
        for real_word in real_line.strip().split():
            if real_word not in real_words:
                real_words[real_word] = 1
            else:
                real_words[real_word] += 1
    for fake_line in fake_lines:
        for fake_word in fake_line.strip().split():
            if fake_word not in fake_words:
                fake_words[fake_word] = 1
            else:
                fake_words[fake_word] += 1
    return real_words, fake_words


def top_words(real_words, fake_words):
    reals = sorted(real_words, key=real_words.get, reverse=True)[:200]
    fakes = sorted(fake_words, key=fake_words.get, reverse=True)[:200]
    top_real=[]
    top_fake=[]
    for realword in reals:
        if realword not in fake_words:
            top_real.append(realword)
    for fakeword in fakes:
        if fakeword not in real_words:
            top_fake.append(fakeword)
    return top_real[:3],top_fake[:3]



if __name__ == '__main__':
    random.seed(0)
    training, validating, testing = generate_sets()
    real_words, fake_words = word_count()
    print(top_words(real_words, fake_words))


