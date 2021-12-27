import torch
import torch.nn as nn
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 weights):
        super(BiLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = weights

        # 1. LAYER EMBEDDING: CREATE A LOOK UP TABLE TO MAP TOKEN-ID TO EMBEDDING
        self.embedding = nn.Embedding.from_pretrained(embeddings=self.weights, freeze=True)

        # 2. LAYER BISLTM: BIDIRECTIONAL LSTM
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True)

        # 3. LAYER FULLY CONNECTED:  map the output of the LSTM into tag space
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, sentence, device):
        """
        Return hidden state so it can be used to initialize the next hidden state
        """
        # Initialize cell state
        h0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim).to(device))
        c0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim).to(device))

        # Forward propagate LSTM
        embedding = self.embedding(sentence).view(len(sentence), 1, -1)  # [seq_len, batch_size, features]
        out, hidden = self.lstm(embedding, (h0, c0))  # out: [seq_len, batch_size, hidden_size]
        out = self.fc(out)

        return out, hidden