from cmath import inf
import nltk
import os
import math

class Bigram:
    def __init__(self):
        self.Unigrams = []
        self.Bigrams = []
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            sentence = '</start> ' + sentence
            self.Unigrams.extend(list(nltk.ngrams(sentence.split(" "), 1)))
            #sentence = '</start> ' + sentence
            self.Bigrams.extend(list(nltk.ngrams(sentence.split(" "), 2)))
        f.close()
        """
        print(len(self.Unigrams))
        print(len(set(self.Unigrams)))
        print(len(self.Bigrams))
        print(len(set(self.Bigrams)))
        """
    
    def fit(self):
        self.count_unigram = {}
        self.prob = {}
        for unigram in self.Unigrams:
            if unigram not in self.count_unigram:
                self.count_unigram[unigram] = 1
            else:
                self.count_unigram[unigram] += 1

        for ngram in self.Bigrams:
            #if ngram[0] != '</start>':
            if ngram not in self.prob:
                self.prob[ngram] = 1 / self.count_unigram[(ngram[0],)]
            else:
                self.prob[ngram] += 1 / self.count_unigram[(ngram[0],)]
        
        print("Training done!")

    def get_train_perplexity(self):
        dev_ngrams = []
        word_count = 0
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            word_count += len(sentence.split(" "))
            sentence = '</start> ' + sentence
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 2)))
        f.close()
        log_prob = 0.0
        for ngram in dev_ngrams:

            if (ngram[0],) not in self.count_unigram:
                ngram = ('<UNK>', ngram[1])
            if (ngram[1],) not in self.count_unigram:
                ngram = (ngram[0], '<UNK>')
            if ngram not in self.prob:
                return(inf)
            log_prob += math.log2(self.prob[ngram])
        print(word_count)
        return 2 ** (-log_prob / word_count)

    def get_dev_perplexity(self):
        dev_ngrams = []
        word_count = 0
        with open(os.path.join(os.getcwd(), "dev_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            word_count += len(sentence.split(" "))
            sentence = '</start> ' + sentence
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 2)))
        f.close()
        log_prob = 0.0
        for ngram in dev_ngrams:

            if (ngram[0],) not in self.count_unigram:
                ngram = ('<UNK>', ngram[1])
            if (ngram[1],) not in self.count_unigram:
                ngram = (ngram[0], '<UNK>')
            if ngram not in self.prob:
                return(inf)
            log_prob += math.log2(self.prob[ngram])

        return 2 ** (-log_prob / word_count)


    def get_test_perplexity(self):
        dev_ngrams = []
        word_count = 0
        with open(os.path.join(os.getcwd(), "test_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            word_count += len(sentence.split(" "))
            sentence = '</start> ' + sentence
            dev_ngrams.extend(list(nltk.ngrams(sentence.split(" "), 2)))
        f.close()
        log_prob = 0.0
        for ngram in dev_ngrams:
            if (ngram[0],) not in self.count_unigram:
                ngram = ('<UNK>', ngram[1])
            if (ngram[1],) not in self.count_unigram:
                ngram = (ngram[0], '<UNK>')
            if ngram not in self.prob:
                return(inf)
            log_prob += math.log2(self.prob[ngram])

        return 2 ** (-log_prob / word_count)