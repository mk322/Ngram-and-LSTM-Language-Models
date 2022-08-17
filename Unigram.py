import nltk
import os
import math
import numpy as np

class Unigram:
    def __init__(self):
        self.Unigrams = []
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop> '
            self.Unigrams.extend(list(nltk.ngrams(sentence.split(), 1)))
        f.close()
    
    def fit(self):
        self.prob = {}
        for ngram in self.Unigrams:
            if ngram not in self.prob:
                self.prob[ngram] = 1 
            else:
                self.prob[ngram] += 1 

        print("Training done!")

    def get_train_perplexity(self):
        size = len(self.Unigrams)
        dev_ngrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split(" "))
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 1)))
        f.close()
        dev_prob = 0
        for ngram in dev_ngrams:
            if ngram not in self.prob:
                dev_prob += math.log2(self.prob[('<UNK>',)] / size)
            else:
                dev_prob += math.log2(self.prob[ngram] / size)
        print(num_words)
        return 2**(-dev_prob / num_words)

    def get_dev_perplexity(self):
        size = len(self.Unigrams)
        dev_ngrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "dev_v2.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split(" "))
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 1)))
        f.close()
        dev_prob = 0
        for ngram in dev_ngrams:
            if ngram not in self.prob:
                ngram = ('<UNK>',)
            dev_prob += math.log2(self.prob[ngram] / size)
        print(num_words)
        return 2**(-dev_prob / num_words)


    def get_test_perplexity(self):
        size = len(self.Unigrams)
        dev_ngrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "test_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split(" "))
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 1)))
        f.close()
        dev_prob = 0
        for ngram in dev_ngrams:
            if ngram not in self.prob.keys():
                ngram = ('<UNK>',)
            dev_prob += math.log(self.prob[ngram] / size, 2)
        print(num_words)
        return 2**(-dev_prob / num_words)