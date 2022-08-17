import matplotlib.pyplot as plt 
from ngram import Smoothed_Ngram
import pandas as pd
import numpy as np
import six  
def main():
    
    smoothed_ngram4 = Smoothed_Ngram(0.4, 0.5, 0.1)
    smoothed_ngram4.fit()
    train_4 = smoothed_ngram4.get_train_perplexity()
    dev_4 = smoothed_ngram4.get_dev_perplexity()
    test_4 = smoothed_ngram4.get_test_perplexity()

    print(train_4)
    print(dev_4)
    print(test_4)

    lstm_train = 63.709
    lstm_dev = 112.495
    lstm_test = 105.097
    
    names = ['Smoothed Trigram', 'LSTM']
    train_val = [round(train_4,3), lstm_train]
    dev_val = [round(dev_4,3), lstm_dev]
    test_val = [round(test_4,3), lstm_test]


    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11, 7.5))
    fig.suptitle('Comparison with LSTM and Smoothed Trigram')
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
    fig.savefig('3.2_compare_ngram.png')
    plt.show()

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
        fig.suptitle('Perplexities of Different Lambdas on the Three Sets', fontsize=16)
        fig.savefig('3.1_table.png')
        return ax

    #render_mpl_table(df, header_columns=0, col_width=2.0)

if __name__ == "__main__":
    main()
