from pandas_ta.utils import weights
import torch
import torch.nn as nn

class LSTMAttention(nn.Module):

    def __init__(self, input_size, hidden_size=64):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        weights = torch.softmax(self.attention(lstm_out), dim=1)

        context = torch.sum(weights * lstm_out, dim=1)

        output = self.fc(context)

        return output