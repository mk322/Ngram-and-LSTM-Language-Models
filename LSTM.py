from cmath import inf
import torch
import numpy as np
import torch.nn as nn
import time
import os
import model
import math
from torch.nn.utils import clip_grad_norm_
from data_utils import Corpus
import matplotlib.pyplot as plt

data = os.path.join(os.getcwd(), 'data')
emsize = 650
nhid = 650
nlayers = 2
batch_size = 20
lr = 5
clip = 0.5
epochs = 50
bptt = 35
dropout = 0.5
tied = True
log_interval = 200
eval_batch_size = 20

device = torch.device('cuda')
corpus = Corpus(data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

model = model.RNNModel(ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
criterion = nn.CrossEntropyLoss()

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

lr = lr
best_val_loss = inf
dev_ppl = []
valid_epoch = 0
try:
    for epoch in range(1, epochs+1):
        valid_epoch += 1
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, np.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            dev_ppl.append(round(np.exp(val_loss),3))
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

train_loss = evaluate(train_data)
dev_loss = evaluate(val_data)


test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, np.exp(test_loss)))
print('=' * 89)

with open('result.txt', 'a') as f:
    for i in dev_ppl:
        print(i, file=f)
    print('====',file=f)
    print(np.exp(train_loss), file=f)
    print(np.exp(dev_loss), file=f)
    print(np.exp(test_loss), file=f)
f.close()

x = [i for i in range(1, valid_epoch)]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 7.5))
fig.suptitle('Development Perplexity against # of Epochs')
ax.plot(x, dev_ppl, 'b-')
fig.savefig('3.22_devppl.png')
plt.show()

