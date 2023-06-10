# Building original GPT model, train it, and generate text

The original GPT model is based on the transformer decoder architecture, the improved GPT-2 and GPT-3 models are based on the transformer encoder architecture. This part one of the tutorial will focus on the original GPT model for simplicity.

We will mainly use PyTorch for building and training the model.

## Define the Model Architecture

First we define the architecture of the GPT model. The GPT model is based on the transformer decoder architecture, which consists of several decoder layers. Each decoder layer has a self-attention mechanism and a position-wise feed-forward network.

```python
import torch
import torch.nn as nn
from torch.nn import Transformer

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GPT, self).__init__()
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_decoder_layers=num_layers)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embed(src)
        output = self.transformer(embedded)
        output = self.fc(output)
        return output
```

In this code, vocab_size is the size of the vocabulary, d_model is the dimension of the embeddings, nhead is the number of attention heads, and num_layers is the number of decoder layers.

## Prepare the Data

Next, we need to prepare the data. The data needs to be tokenized and converted into tensors. We will not cover this step in detail here, but PyTorch provides utilities for loading and preprocessing data that you can use.

## Train the Model

Once we have defined the model architecture and prepared the data, we can train the model using gradient descent. The loss function for a language model is usually the cross-entropy loss.

```python
# Assume we have a DataLoader `data_loader` that loads the training data
model = GPT(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
model = model.to('cuda')  # Assume we're using a GPU
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (src, tgt) in enumerate(data_loader):
        src = src.to('cuda')
        tgt = tgt.to('cuda')
        output = model(src)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

In this code, num_epochs is the number of epochs to train for, and data_loader is a DataLoader that loads the training data.

## Evaluate the Model

Finally, after the model has been trained, we can evaluate it using a metric like perplexity. Perplexity is defined as exp(loss), where loss is the cross-entropy loss.

```python
model.eval()
total_loss = 0
with torch.no_grad():
    for i, (src, tgt) in enumerate(data_loader):
        src = src.to('cuda')
        tgt = tgt.to('cuda')
        output = model(src)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        total_loss += loss.item()

avg_loss = total_loss / len(data_loader)
perplexity = torch.exp(torch.tensor(avg_loss))
```

This code calculates the average loss over all the data in data_loader and then computes the perplexity.

## Generate Text

Once the model is trained, you can use it to generate text.

```python

def generate(model, start_sentence, max_length):
    model.eval()
    sentence = start_sentence
    with torch.no_grad():
        for _ in range(max_length - len(start_sentence)):
            input = torch.tensor([sentence], dtype=torch.long).to('cuda')
            output = model(input)
            next_word = output.argmax(dim=2)[:, -1].item()
            sentence.append(next_word)
    return sentence
```

This generate function takes as input a model, a start_sentence (which is a list of token IDs), and a max_length. It generates text by repeatedly predicting the next word based on the current sentence.

Please note this guide provides a high-level overview and simplified version of a GPT model. The actual model is significantly more complex and involves additional components such as positional encoding, layer normalization, and masked self-attention
