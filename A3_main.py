from bitarray import test
import matplotlib.pyplot as plt
from torch import half 
from Unigram import Unigram
from Bigram_1 import Bigram
from smoothed_ngram import Smoothed_Ngram
from Trigram import Trigram
import pandas as pd
import numpy as np
import six  
def main():
    """
    result = []
    with open(os.path.join(os.getcwd(),'1b_benchmark.train.tokens'), encoding="UTF8") as f:
        file = f.read()
        sentences = file.split("\n")
        l = file.split()
        f.close()
    for sentence in sentences:
        result.append(sentence.split())

    #print(result)

    dic = Counter(l)

    for sentence in result:
        for i in range(len(sentence)):
            if dic[sentence[i]] < 3:
                sentence[i] = "<UNK>"

    #print(len(set()))

    with open(os.path.join(os.getcwd(), "train_data.txt"), "a", encoding='UTF8') as t:
        for sentence in result:
            for word in sentence:
                print(word, file=t, end=' ')
            print("",file=t)
        t.close()
    """
    """
    unigram = Unigram()
    unigram.fit()
    print('Uigrams: ')
    uni_train = round(unigram.get_train_perplexity(), 3)
    uni_dev = round(unigram.get_dev_perplexity(), 3)
    uni_test = round(unigram.get_test_perplexity(), 3)
    print('Perplexity for train set: ' + str(uni_train))
    print('Perplexity for valid set: ' + str(uni_dev))
    print('Perplexity for test set: ' + str(uni_test))
    
    bigram = Bigram()
    bigram.fit()
    print('Bigrams: ')
    bi_train = round(bigram.get_train_perplexity(), 3)
    bi_dev = round(bigram.get_dev_perplexity(), 3)
    bi_test = round(bigram.get_test_perplexity(), 3)
    print('Perplexity for train set: ' + str(bi_train))
    print('Perplexity for valid set: ' + str(bi_dev))
    print('Perplexity for test set: ' + str(bi_test))
    
    trigram =  Trigram()
    trigram.fit()
    print('Trigrams: ')
    tri_train = round(trigram.get_train_perplexity(), 3)
    tri_dev = round(trigram.get_dev_perplexity(), 3)
    tri_test = round(trigram.get_test_perplexity(), 3)
    print('Perplexity for train set: ' + str(tri_train))
    print('Perplexity for valid set: ' + str(tri_dev))
    print('Perplexity for test set: ' + str(tri_test))
    

    names = ['Unigram', 'Bigram', 'Trigram']
    train_val = [uni_train, bi_train, tri_train]
    dev_val = [892.247, 999999, 999999]
    test_val = [896.499, 9999999999, 9999999999]


    figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11, 7.5))
    figure.suptitle('Perplexities of Three Ngrams')


    ax1.bar(names, train_val)
    ax1.set_ylim([0, 3000])
    for index, value in enumerate(train_val):
        ax1.text(index-0.32, value, str(value))
    ax1.title.set_text('Train Set')


    ax2.bar(names, dev_val)
    ax2.set_ylim([0, 3000])

    ax2.text(0-0.32, 892.247, str(892.247))
    ax2.text(1-0.1, 2800, "Inf")
    ax2.text(2-0.1, 2800, "Inf")
    ax2.title.set_text('Dev Set')


    plt.ylim([0, 3000])
    ax3.bar(names, test_val)
    ax3.set_ylim([0, 3000])
    ax3.text(0-0.32, 896.499, str(896.499))
    ax3.text(1-0.1, 2800, "Inf")
    ax3.text(2-0.1, 2800, "Inf")
    ax3.title.set_text('Test Set')

    figure.tight_layout(pad=1.0)
    figure.savefig("3.1.png")
    plt.show()
    
    
    smoothed_ngram1 = Smoothed_Ngram(0.1, 0.3, 0.6)
    smoothed_ngram1.fit()
    train_1 = smoothed_ngram1.get_train_perplexity()
    dev_1 = smoothed_ngram1.get_dev_perplexity()
    #print('Perplexity for train set: ' + str(smoothed_ngram.get_train_perplexity()))
    #print('Perplexity for valid set: ' + str(smoothed_ngram.get_dev_perplexity()))
    #print('Perplexity for test set: ' + str(smoothed_ngram.get_test_perplexity()))

    smoothed_ngram2 = Smoothed_Ngram(0.8, 0.1, 0.1)
    smoothed_ngram2.fit()
    train_2 = smoothed_ngram2.get_train_perplexity()
    dev_2 = smoothed_ngram2.get_dev_perplexity()


    smoothed_ngram3 = Smoothed_Ngram(0.1, 0.8, 0.1)
    smoothed_ngram3.fit()
    train_3 = smoothed_ngram3.get_train_perplexity()
    dev_3 = smoothed_ngram3.get_dev_perplexity()
    test_3 = smoothed_ngram3.get_test_perplexity()
    
    smoothed_ngram4 = Smoothed_Ngram(0.2, 0.7, 0.1)
    smoothed_ngram4.fit()
    train_4 = smoothed_ngram4.get_train_perplexity()
    dev_4 = smoothed_ngram4.get_dev_perplexity()
    test_4 = smoothed_ngram4.get_test_perplexity()
    print(train_4)
    print(dev_4)
    print(test_4)

    smoothed_ngram5 = Smoothed_Ngram(0.2, 0.7, 0.1, half=True)
    smoothed_ngram5.fit()
    train_5 = smoothed_ngram5.get_train_perplexity()
    dev_5 = smoothed_ngram5.get_dev_perplexity()
    test_5 = smoothed_ngram5.get_test_perplexity()
    print(train_5)
    print(dev_5)
    print(test_5)
    """
    df = pd.DataFrame()
    df['Lambda'] = ['(0.1, 0.1, 0.8)','(0.1, 0.8, 0.1)', '(0.8, 0.1, 0.1)', '(0.1, 0.3, 0.6)']
    df['Train Set'] = [9.330, 7.482, 47.040, 11.151]
    df['Validation Set'] = [482.846, 204.580, 373.676, 352.234]
    #df['Test Set'] = [480.943, 203.881, 374.491, 351.007]


    #smoothed_ngram_half = Smoothed_Ngram(0.1, 0.8, 0.1, appear_1=True)
    #smoothed_ngram_half.fit()
    #train_h = smoothed_ngram_half.get_train_perplexity()
    #dev_h = smoothed_ngram_half.get_dev_perplexity()
    #test_h = smoothed_ngram_half.get_test_perplexity()

    
    """
    names = ['Use Half Data', 'Use All Data']
    train_val = [23.574, 28.0901]
    dev_val = [286.885, 292.678]
    test_val = [286.172, 292.299]


    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11, 7.5))
    fig.suptitle('Report of Different Amount of Train Data')
    ax1.bar(names, train_val)
    ax2.bar(names, dev_val)
    ax3.bar(names, test_val)
    for index, value in enumerate(train_val):
        ax1.text(index-0.2, value, str(value))
    ax1.title.set_text('Train Set')
    for index, value in enumerate(dev_val):
        ax2.text(index-0.2, value, str(value))
    ax2.title.set_text('Validation Set')

    for index, value in enumerate(test_val):
        ax3.text(index-0.2, value, str(value))
    ax3.title.set_text('Test Set')
    fig.savefig('3.2_half_.png')
    plt.show()
    """
    def render_mpl_table(data, col_width=8, row_height=1, font_size=13,
                        header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                        bbox=[0, 0, 1, 1], header_columns=0,
                        ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        fig.suptitle('Perplexities of Different Lambdas', fontsize=16)
        fig.savefig('3.1_table.png')
        return ax
       

    render_mpl_table(df, header_columns=0, col_width=2.0)

if __name__ == "__main__":
    main()
