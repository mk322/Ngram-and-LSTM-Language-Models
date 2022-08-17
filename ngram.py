import nltk
import os
import math

class Smoothed_Ngram:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.Unigrams = []
        self.Bigrams = []
        self.Trigrams = []
        self.Unigrams2 = []
        self.Bigrams2 = []
        self.vocab_size = 0
        with open(os.path.join(os.getcwd(), 'data','train.txt'), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            self.vocab_size += len(sentence.split(" "))
            sentence = '</start> ' + sentence
            self.Unigrams2.extend(list(nltk.ngrams(sentence.split(" "), 1)))
            self.Bigrams2.extend(list(nltk.ngrams(sentence.split(" "), 2)))
            sentence = '</start> ' + sentence
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
        self.count_bigram2 = {}
        self.count_unigram2 = {}
        for unigram in self.Unigrams:
            if unigram not in self.count_unigram:
                self.count_unigram[unigram] = 1
            else:
                self.count_unigram[unigram] += 1
        
        for unigram in self.Unigrams2:
            if unigram not in self.count_unigram2:
                self.count_unigram2[unigram] = 1
            else:
                self.count_unigram2[unigram] += 1

        for bigram in self.Bigrams:
            if bigram not in self.count_bigram:
                self.count_bigram[bigram] = 1
            else:
                self.count_bigram[bigram] += 1
        
        for bigram in self.Bigrams2:
            if bigram not in self.count_bigram2:
                self.count_bigram2[bigram] = 1
            else:
                self.count_bigram2[bigram] += 1


        for trigram in self.Trigrams:
            if trigram not in self.count_trigram:
                self.count_trigram[trigram] = 1
            else:
                self.count_trigram[trigram] += 1
      
        print("Training done!")

    def get_train_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), 'data', "train.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:
            p1 = 0
            p2 = 0
            p3 = 0    
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')

            if trigram in self.count_trigram:
                p3 = self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])]

            if (trigram[1],trigram[2]) in self.count_bigram:
                p2 = self.count_bigram2[(trigram[1],trigram[2])] / self.count_unigram2[(trigram[1],)]

            if (trigram[2],) in self.count_unigram:
                p1 = self.count_unigram2[(trigram[2],)] / self.vocab_size

            log_prob += math.log2(p1*self.l1 + p2*self.l2 + p3*self.l3)
        print(num_words)
        return 2**(-log_prob / num_words)

    def get_dev_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), 'data', "valid.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:
            p1 = 0
            p2 = 0
            p3 = 0    
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')
            if trigram in self.count_trigram:
                p3 = self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])]
            if (trigram[1],trigram[2]) in self.count_bigram:
                p2 = self.count_bigram2[(trigram[1],trigram[2])] / self.count_unigram2[(trigram[1],)]
            if (trigram[2],) in self.count_unigram:
                p1 = self.count_unigram[(trigram[2],)] / self.vocab_size
            log_prob += math.log2(p1*self.l1 + p2*self.l2 + p3*self.l3)

        return 2**(-log_prob / num_words)


    def get_test_perplexity(self):
        dev_trigrams = []
        num_words = 0
        with open(os.path.join(os.getcwd(), 'data', "test.txt"), encoding="UTF8") as f:
            sentences = f.read().split("\n")
        for sentence in sentences:
            sentence += '</stop>'
            num_words += len(sentence.split())
            sentence = '</start> </start> ' + sentence
            dev_trigrams.extend(list(nltk.ngrams(sentence.split(' '), 3)))
        f.close()
        log_prob = 0.0
        for trigram in dev_trigrams:
            p1 = 0
            p2 = 0
            p3 = 0    
            if (trigram[0],) not in self.count_unigram:
                trigram = ('<UNK>', trigram[1], trigram[2])
            if (trigram[1],) not in self.count_unigram:
                trigram = (trigram[0], '<UNK>', trigram[2])
            if (trigram[2],) not in self.count_unigram:
                trigram = (trigram[0], trigram[1], '<UNK>')
            if trigram in self.count_trigram:
                p3 = self.count_trigram[trigram] / self.count_bigram[(trigram[0], trigram[1])]
            if (trigram[1],trigram[2]) in self.count_bigram:
                p2 = self.count_bigram2[(trigram[1],trigram[2])] / self.count_unigram2[(trigram[1],)]
            if (trigram[2],) in self.count_unigram:
                p1 = self.count_unigram[(trigram[2],)] / self.vocab_size
            log_prob += math.log2(p1*self.l1 + p2*self.l2 + p3*self.l3)

        return 2**(-log_prob / num_words)
