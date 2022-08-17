from cmath import inf
import nltk
import os
import math

class Trigram:
    def __init__(self):
        self.Unigrams = []
        self.Bigrams = []
        self.Trigrams = []
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            sentence = '</start> </start> ' + sentence
            self.Unigrams.extend(list(nltk.ngrams(sentence.split(" "), 1)))
            self.Bigrams.extend(list(nltk.ngrams(sentence.split(" "), 2)))
            self.Trigrams.extend(list(nltk.ngrams(sentence.split(" "), 3)))
        f.close()
        """
        print(len(self.Unigrams))
        print(len(set(self.Unigrams)))
        print(len(self.Bigrams))
        print(len(set(self.Bigrams)))
        print(len(self.Trigrams))
        print(len(set(self.Trigrams)))
        """    
    
    def fit(self):
        self.count_unigram = {}
        self.count_bigram = {}
        self.count_trigram = {}
        for unigram in self.Unigrams:
            if unigram not in self.count_unigram:
                self.count_unigram[unigram] = 1
            else:
                self.count_unigram[unigram] += 1

        for bigram in self.Bigrams:
            if bigram not in self.count_bigram:
                self.count_bigram[bigram] = 1
            else:
                self.count_bigram[bigram] += 1

        for trigram in self.Trigrams:
            if trigram not in self.count_trigram:
                self.count_trigram[trigram] = 1
            else:
                self.count_trigram[trigram] += 1

        print("Training done!")

    def get_train_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "train_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:   
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')
            if trigram not in self.count_trigram:
                return inf
            log_prob += math.log2(self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])])
        print(num_words)
        return 2**(-log_prob / num_words)

    def get_dev_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "dev_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:   
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')
            if trigram not in self.count_trigram:
                return inf
            log_prob += math.log2(self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])])

        return 2**(-log_prob / num_words)

    def get_test_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), "test_data.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:   
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')
            if trigram not in self.count_trigram:
                return inf
            log_prob += math.log2(self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])])

        return 2**(-log_prob / num_words)